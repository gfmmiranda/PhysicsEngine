import jax
import jax.numpy as jnp
import numpy as np
from src.core import PDESolver

# --- 1. THE JIT-COMPILED KERNEL (Runs on GPU) ---
# This function is "Pure": Inputs -> Outputs. No 'self', no side effects.
@jax.jit
def wave_step_kernel(
    u_curr, u_prev, 
    c, dt, inv_dx_sq, # Constants (scalars or small arrays)
    wall_indices, air_indices, dn_vals, R_vals # Boundary Arrays
):
    # A. Laplacian (Vectorized Stencil)
    # We use jnp.roll for neighbor access. It's efficient in XLA.
    # u(x+1) + u(x-1) - 2u(x)
    d2x = (jnp.roll(u_curr, -1, axis=0) - 2*u_curr + jnp.roll(u_curr, 1, axis=0)) * inv_dx_sq[0]
    d2y = (jnp.roll(u_curr, -1, axis=1) - 2*u_curr + jnp.roll(u_curr, 1, axis=1)) * inv_dx_sq[1]
    lap = d2x + d2y
    
    # B. Update Interior
    u_next = 2*u_curr - u_prev + (c * dt)**2 * lap
    
    # C. Apply Boundaries (Parallel)
    # 1. Gather Data for ALL boundary pixels at once
    u_wall_n   = u_curr[wall_indices]
    u_wall_nm1 = u_prev[wall_indices]
    u_air_n    = u_curr[air_indices]
    
    # Get the FUTURE air neighbor (from the u_next we just calculated)
    u_air_next = u_next[air_indices]
    
    # 2. Physics Math (Mur + Neumann)
    lam = (c * dt) / dn_vals
    
    # Absorbing Limit
    coeff = (lam - 1.0) / (lam + 1.0)
    val_absorb = u_air_n + coeff * (u_air_next - u_wall_n)
    
    # Hard Limit (Ghost Point)
    val_hard = (2.0 * u_wall_n - u_wall_nm1 + 
                (lam**2) * (2.0 * u_air_n - 2.0 * u_wall_n))
    
    # Mix based on R
    val_boundary = (R_vals * val_hard) + ((1.0 - R_vals) * val_absorb)
    
    # 3. Write back (Immutable update)
    u_next = u_next.at[wall_indices].set(val_boundary)
    
    return u_next, u_curr

# --- 2. THE CLASS (Manages Data) ---
class WaveJAX:
    def __init__(self, domain, c=343.0, dt=None, boundary_type='robin'):
        self.domain = domain
        self.c = c
        self.t = 0.0
        
        # Setup Grid
        # Pre-calc 1/dx^2 for speed
        self.inv_dx_sq = jnp.array([1.0/d**2 for d in domain.ds])
        
        # CFL Condition
        inv_sq_sum = np.sum(1.0 / domain.ds**2)
        cfl_limit = 1.0 / (c * np.sqrt(inv_sq_sum))
        self.dt = 0.99 * cfl_limit if dt is None or dt >= cfl_limit else dt
        
        # Compile Geometry for GPU
        self._compile_gpu_data()
        
        # Initialize State on GPU
        # We start with zeros. If you have initial conditions, map them here.
        self.u_curr = jnp.zeros(domain.N)
        self.u_prev = jnp.zeros(domain.N)
        
        print("Compiling JAX kernel (Warmup)...")
        _ = self.step() # Trigger JIT compilation
        self.t = 0.0    # Reset time after warmup
        print("Model compiled and ready on GPU.")

    def _compile_gpu_data(self):
        """Converts the domain's sparse maps into dense GPU arrays."""
        if not hasattr(self.domain, 'neumann_map'):
            self.domain.generate_neumann_map()
            
        wall_idxs = []
        air_idxs = []
        dns = []
        rs = []
        
        # Read the CPU material map
        R_map = self.domain.materials
        
        for (wall, air) in self.domain.neumann_map:
            # Determine axis (normal direction)
            axis = 0
            for dim in range(self.domain.ndim):
                if wall[dim] != air[dim]:
                    axis = dim
                    break
            
            wall_idxs.append(wall)
            air_idxs.append(air)
            dns.append(self.domain.ds[axis])
            rs.append(R_map[wall])
            
        # Convert to JAX Indices: (Array[X_coords], Array[Y_coords])
        # We transpose the list of tuples to get [[x1, x2...], [y1, y2...]]
        self.wall_indices = tuple(jnp.array(np.array(wall_idxs).T))
        self.air_indices  = tuple(jnp.array(np.array(air_idxs).T))
        
        self.dn_vals = jnp.array(dns)
        self.R_vals  = jnp.array(rs)

    def step(self):
        # 1. Add Dynamic Sources (CPU -> GPU)
        # For a few sources, adding on CPU before transfer is fine.
        # For optimized runs, we'd eventually move source logic to GPU too.
        src_field = np.zeros(self.domain.N)
        for s in self.domain.sources:
            # Simple addition to the correct pixel
            src_field[s.grid_idx] += s.value(self.t)
            
        if np.any(src_field):
            # Move source term to GPU and apply
            # (Force * dt^2) is the displacement contribution
            term = jnp.array(src_field) * (self.dt**2)
            self.u_curr = self.u_curr + term

        # 2. Run Kernel
        self.u_curr, self.u_prev = wave_step_kernel(
            self.u_curr, self.u_prev,
            self.c, self.dt, self.inv_dx_sq,
            self.wall_indices, self.air_indices, self.dn_vals, self.R_vals
        )
        
        self.t += self.dt
        
        # 3. Listeners (GPU -> CPU Readback)
        for l in self.domain.listeners:
             # accessing [l.grid_idx] triggers a device-to-host copy for that single float
             val = self.u_curr[l.grid_idx]
             l.record(self.t, float(val))