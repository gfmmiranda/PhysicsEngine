import numpy as np

class PhysicsSolver:
    """
    Base class for 2D finite difference solvers. 
    Handles domain binding, boundary conditions, and spatial derivatives.
    """
    def __init__(self, domain, boundary_type='dirichlet'):
        self.domain = domain
        self.boundary_type = boundary_type
        self.dim = 2  # Fixed for 2D solvers
        
        # Pre-compute Neumann map if needed
        if self.boundary_type == 'neumann':
            if not hasattr(self.domain, 'neumann_map'):
                self.domain.generate_neumann_map()

    def apply_boundary_conditions(self, u):
        """Enforces Dirichlet or Neumann BCs on the given field u."""
        if self.boundary_type == 'dirichlet':
            u[~self.domain.mask] = 0.0

        elif self.boundary_type == 'neumann':
            u[~self.domain.mask] = 0.0
            for (wall_idx, air_idx) in self.domain.neumann_map:
                u[wall_idx] = u[air_idx]

    def laplacian(self, u):
        """Standard 5-point stencil finite difference Laplacian."""
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.domain.dx**2 +
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / self.domain.dy**2
        )
        return lap

    def step(self):
        """Must be implemented by child classes."""
        raise NotImplementedError("Each solver must implement its own time-stepping logic.")