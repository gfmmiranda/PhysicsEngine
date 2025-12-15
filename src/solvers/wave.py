import numpy as np
from src.core import PDESolver

class Wave(PDESolver):
    def __init__(
            self, 
            domain, 
            initial_u, 
            initial_ut, 
            dt=None, 
            c=1.0, 
            boundary_type='dirichlet',
            alpha=0.0
            ):
        
        # If user sets alpha > 0 and boundary_type is dirichlet, override to robin
        if alpha > 0.0 and boundary_type == 'dirichlet':
            print("Alpha > 0 with Dirichlet BCs detected. Switching to Robin BCs for absorption.")
            boundary_type = 'robin'
        
        # Pass generic args to parent
        super().__init__(domain, boundary_type)
        self.name = 'Wave' 
        
        # Wave specific arguments
        self.c = c
        self.phi = initial_u
        self.psi = initial_ut
        self.alpha = alpha
        
        # CFL Condition: dt <= dx / (c * sqrt(2))
        inv_sq_sum = np.sum(1.0 / self.domain.ds**2)
        cfl_limit = 1.0 / (c * np.sqrt(inv_sq_sum))
        self.dt = 0.99 * cfl_limit if dt is None or dt >= cfl_limit else dt

        # Boundary Conditions Initialization
        self.boundary_physics_map = []
        
        if self.boundary_type in ['neumann', 'robin']:
            self._compile_boundary_physics()

        self.initialize_state()

    def _compile_boundary_physics(self):
        """
        Iterates over the Domain's geometry map and pre-calculates 
        the Robin 'beta' term for every single boundary point.
        """

        count = 0
        for (wall_idx, air_idx) in self.domain.neumann_map:
            # A. Detect Normal Axis (to handle anisotropic dx)
            axis = 0
            for dim in range(self.domain.ndim):
                if wall_idx[dim] != air_idx[dim]:
                    axis = dim
                    break
            
            # B. Get spacing normal to this specific wall
            dn = self.domain.ds[axis]
            
            # C. Calculate Physics Coefficient (Beta)
            beta = (self.alpha * dn) / (2 * self.dt)
            count += 1
            
            self.boundary_physics_map.append((wall_idx, air_idx, beta))

    def initialize_state(self):
        """Initialize u_prev and u_curr based on initial conditions."""
        self.u_prev = self.phi(*self.domain.grids)
        self.u_curr = self.u_prev + self.dt * self.psi(*self.domain.grids)

        self.apply_boundary_conditions(self.u_prev)
        self.apply_boundary_conditions(self.u_curr)

    def apply_boundary_conditions(self, u):
        # 1. Dirichlet (Mask defaults to 0)
        u[~self.domain.mask] = 0.0

        # 2. Advanced Conditions (Neumann / Robin)
        # We use our pre-compiled list. No if-checks inside the loop.
        for (wall_idx, air_idx, beta) in self.boundary_physics_map:
            
            val_air = u[air_idx]
            if beta == 0:
                # Optimized Neumann (Hard Wall): u_wall = u_air
                u[wall_idx] = val_air
            else:
                # Robin (Absorbing): Uses history (u_prev)
                val_prev = self.u_prev[wall_idx]
                u[wall_idx] = (val_air + beta * val_prev) / (1 + beta)

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
