import numpy as np
from src.core import PDESolver

class Wave(PDESolver):
    def __init__(
            self, 
            domain, 
            initial_u = lambda x, y: np.zeros_like(x), 
            initial_ut = lambda x, y: np.zeros_like(x), 
            dt=None, 
            c=1.0, 
            boundary_type='dirichlet',
            R=1.0
            ):
        
        # If user sets alpha > 0 and boundary_type is dirichlet, override to robin
        if R < 1.0 and boundary_type == 'dirichlet':
            print("R < 1.0 with Dirichlet BCs detected. Switching to Robin BCs for absorption.")
            boundary_type = 'robin'
        
        # Pass generic args to parent
        super().__init__(domain, boundary_type)
        self.name = 'Wave' 
        
        # Wave specific arguments
        self.c = c
        self.phi = initial_u
        self.psi = initial_ut
        self.R = R
        
        # CFL Condition: dt <= dx / (c * sqrt(2))
        inv_sq_sum = np.sum(1.0 / self.domain.ds**2)
        cfl_limit = 1.0 / (c * np.sqrt(inv_sq_sum))
        self.dt = 0.99 * cfl_limit if dt is None or dt >= cfl_limit else dt

        # Boundary Conditions Initialization
        self.boundary_physics_map = []
        if self.boundary_type in ['neumann', 'robin']:
            self._compile_boundary_physics()

        self.initialize_state()

    def initialize_state(self):
        """Initialize u_prev and u_curr based on initial conditions."""
        self.u_prev = self.phi(*self.domain.grids)
        self.u_curr = self.u_prev + self.dt * self.psi(*self.domain.grids)

        self.apply_boundary_conditions(self.u_prev)
        self.apply_boundary_conditions(self.u_curr)
    

    def apply_boundary_conditions(self, u):
        # 1. Dirichlet (Mask defaults to 0)
        u[~self.domain.mask] = 0.0

        # 2. Partial Absorption (Robin / Impedance)
        for (wall_idx, air_idx, dn, R) in self.boundary_physics_map:
            
            # --- Setup ---
            dt = self.dt
            c = self.c
            lam = (c * dt) / dn  # Courant number
            
            # --- Gather Data ---
            u_wall_n   = self.u_curr[wall_idx]    # Wall at current time (t)
            u_wall_nm1 = self.u_prev[wall_idx]    # Wall at previous time (t-1)
            u_air_n    = self.u_curr[air_idx]     # Air neighbor at current time (t)
            
            # CRITICAL FIX: Get the Air Neighbor at the NEXT time (t+1)
            u_air_next = u[air_idx] 

            # --- Calculate Extremes ---
            
            # A. Absorbing Limit (Corrected Mur 1st Order)
            # Formula: u_wall_next = u_air_curr + (lam-1)/(lam+1) * (u_air_next - u_wall_curr)
            # This correctly "transports" the wave out of the domain.
            coeff_absorb = (lam - 1.0) / (lam + 1.0)
            val_absorb = u_air_n + coeff_absorb * (u_air_next - u_wall_n)
            
            # B. Hard Limit (Corrected 2nd Order Neumann)
            # We use the ghost point method (u_ghost = u_air) plugged into the wave equation.
            # This ensures the wall follows the standard wave physics (stable).
            val_hard = (2.0 * u_wall_n - u_wall_nm1 + 
                        (lam**2) * (2.0 * u_air_n - 2.0 * u_wall_n))
            
            # --- Mix ---
            # Linear interpolation between Hard (R=1) and Absorbing (R=0)
            u[wall_idx] = (R * val_hard) + ((1.0 - R) * val_absorb)

    def _compile_boundary_physics(self):
        """
        Iterates over the Domain's geometry map and stores the
        Reflection Coefficient (R) and spacing (dn) for every boundary point.
        """
        for (wall_idx, air_idx) in self.domain.neumann_map:
            # A. Detect Normal Axis (Keep this! It's expensive to do in the loop)
            axis = 0
            for dim in range(self.domain.ndim):
                if wall_idx[dim] != air_idx[dim]:
                    axis = dim
                    break
            
            # B. Get spacing normal to this specific wall
            dn = self.domain.ds[axis]
            
            # C. Get Reflection Coefficient for this wall
            val_R = self.R

            # D. Store 'dn' too! This saves us from finding 'axis' every single time step.
            self.boundary_physics_map.append((wall_idx, air_idx, dn, val_R))

    def step(self):
        # 1. Compute Physics
        lap = self.laplacian(self.u_curr)
        u_next = (2 * self.u_curr - self.u_prev + 
                    (self.c * self.dt)**2 * (lap + self.active_source_field()))

        # 2. Enforce Boundaries
        self.apply_boundary_conditions(u_next)

        # 3. Update State
        self.u_prev, self.u_curr = self.u_curr, u_next
        self.t += self.dt

        # 4. Record Data
        for listener in self.listeners:
            listener.record(self.t, self.u_curr)
