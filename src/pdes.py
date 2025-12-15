import numpy as np
from src.pdesolver import PDESolver

class Wave(PDESolver):
    def __init__(self, domain, initial_u, initial_ut, dt=None, c=1.0, boundary_type='dirichlet'):
        # Pass generic args to parent
        super().__init__(domain, boundary_type)
        self.name = 'Wave' 
        
        # Wave specific arguments
        self.c = c
        self.phi = initial_u
        self.psi = initial_ut
        
        # CFL Condition: dt <= dx / (c * sqrt(2))
        inv_sq_sum = np.sum(1.0 / self.domain.ds**2)
        cfl_limit = 1.0 / (c * np.sqrt(inv_sq_sum))
        self.dt = 0.99 * cfl_limit if dt is None or dt >= cfl_limit else dt
        
        self.initialize_state()

    def initialize_state(self):
        self.u_prev = self.phi(*self.domain.grids)
        self.u_curr = self.u_prev + self.dt * self.psi(*self.domain.grids)

        self.apply_boundary_conditions(self.u_prev)
        self.apply_boundary_conditions(self.u_curr)

    def step(self):
        # 1. Compute Physics
        lap = self.laplacian(self.u_curr)
        u_next = (2 * self.u_curr - self.u_prev + 
                  (self.c * self.dt)**2 * lap)

        # 2. Enforce Boundaries
        self.apply_boundary_conditions(u_next)

        # 3. Update State
        self.u_prev, self.u_curr = self.u_curr, u_next


class Heat(PDESolver):
    def __init__(self, domain, initial_u, dt=None, k=1.0, boundary_type='dirichlet'):
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

    def step(self):
        # 1. Compute Physics
        lap = self.laplacian(self.u_curr)
        u_next = self.u_curr + self.k * self.dt * lap

        # 2. Enforce Boundaries
        self.apply_boundary_conditions(u_next)

        # 3. Update State
        self.u_curr = u_next