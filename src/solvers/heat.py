import numpy as np
from src.core import PDESolver

class Heat(PDESolver):
    def __init__(
            self, 
            domain, 
            initial_u, 
            dt=None, 
            k=1.0, 
            boundary_type='dirichlet'
            ):
        
        # Pass generic args to parent
        super().__init__(domain, boundary_type)
        self.name = 'Heat'
        
        # Diffusion specific arguments
        self.k = k
        self.phi = initial_u
        
        # --- N-Dimensional Stability Condition ---
        # 1. Calculate Sum of Inverse Squares
        inv_sq_sum = np.sum(1.0 / self.domain.ds**2)
        
        # 2. Calculate limit: 1 / (2 * k * Sum)
        # Note: In 2D isotropic, Sum = 2/dx^2, so this becomes dx^2 / 4k
        stability_limit = 1.0 / (2 * k * inv_sq_sum)
        
        # 3. Apply safety factor
        self.dt = 0.99 * stability_limit if dt is None or dt >= stability_limit else dt
        
        self.initialize_state()
    
    def initialize_state(self):
        self.u_curr = self.phi(*self.domain.grids)
        self.apply_boundary_conditions(self.u_curr)

    def apply_boundary_conditions(self, u):
        # 1. Dirichlet (Fixed Temperature walls = 0)
        u[~self.domain.mask] = 0.0

        # 2. Neumann (Insulated walls)
        # Heat equation Neumann is simple: Slope = 0 -> u_wall = u_air
        if self.boundary_type == 'neumann':
            for (wall_idx, air_idx) in self.domain.neumann_map:
                u[wall_idx] = u[air_idx]

    def step(self):
        # 1. Compute Physics
        lap = self.laplacian(self.u_curr)
        u_next = self.u_curr + self.dt * (self.k * lap + self.active_source_field())

        # 2. Enforce Boundaries
        self.apply_boundary_conditions(u_next)

        # 3. Update State
        self.u_curr = u_next