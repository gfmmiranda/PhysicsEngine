import numpy as np
from src.base_solver import PhysicsSolver

class Wave2D(PhysicsSolver):
    def __init__(self, domain, initial_u, initial_ut, dt=None, c=1.0, boundary_type='dirichlet'):
        # Pass generic args to parent
        super().__init__(domain, boundary_type)
        self.name = 'Wave2D' 
        
        # Wave specific arguments
        self.c = c
        self.phi = initial_u
        self.psi = initial_ut
        
        # CFL Condition: dt <= dx / (c * sqrt(2))
        stability_limit = (domain.dx / (c * np.sqrt(2)))
        self.dt = 0.99 * stability_limit if dt is None or dt >= stability_limit else dt
        
        self.initialize_state()

    def initialize_state(self):
        self.u_prev = self.phi(self.domain.X, self.domain.Y)
        self.u_curr = self.u_prev + self.dt * self.psi(self.domain.X, self.domain.Y)

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


class Heat2D(PhysicsSolver):
    def __init__(self, domain, initial_u, dt=None, k=1.0, boundary_type='dirichlet'):
        # Pass generic args to parent
        super().__init__(domain, boundary_type)
        self.name = 'Heat2D'
        
        # Diffusion specific arguments
        self.k = k
        self.phi = initial_u
        
        # Stability: dt <= dx^2 / 4k
        stability_limit = (domain.dx ** 2) / (4 * k)
        self.dt = 0.99 * stability_limit if dt is None or dt >= stability_limit else dt
        
        self.initialize_state()
    
    def initialize_state(self):
        self.u_curr = self.phi(self.domain.X, self.domain.Y)
        self.apply_boundary_conditions(self.u_curr)

    def step(self):
        # 1. Compute Physics
        lap = self.laplacian(self.u_curr)
        u_next = self.u_curr + self.k * self.dt * lap

        # 2. Enforce Boundaries
        self.apply_boundary_conditions(u_next)

        # 3. Update State
        self.u_curr = u_next