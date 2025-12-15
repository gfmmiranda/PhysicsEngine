import numpy as np

class PDESolver:
    """
    Base class for 2D finite difference solvers. 
    Handles domain binding, boundary conditions, and spatial derivatives.
    """
    def __init__(self, domain, boundary_type='dirichlet'):
        self.domain = domain
        self.boundary_type = boundary_type
        
        # Pre-compute Neumann map if needed
        if self.boundary_type == 'neumann':
            if not hasattr(self.domain, 'neumann_map'):
                self.domain.generate_neumann_map()

    def apply_boundary_conditions(self, u):
        """Enforces Dirichlet or Neumann BCs on the given field u."""
        if not hasattr(self, 'ndim'):
            self.ndim = u.ndim

        if self.boundary_type == 'dirichlet':
            u[~self.domain.mask] = 0.0

        elif self.boundary_type == 'neumann':
            u[~self.domain.mask] = 0.0
            for (wall_idx, air_idx) in self.domain.neumann_map:
                u[wall_idx] = u[air_idx]

    def laplacian(self, u):
        """Standard 5-point stencil finite difference Laplacian."""
        if not hasattr(self, 'ndim'):
            self.ndim = u.ndim

        interior_slice = tuple([slice(1,-1)]*self.ndim)

        lap = np.zeros_like(u)
        for axis in range(self.ndim):
            # Create slices for next, current, and previous along the given axis
            sl_next = list(interior_slice)
            sl_curr = list(interior_slice)
            sl_prev = list(interior_slice)

            # Update the slices for the current axis
            sl_next[axis] = slice(2, None)   # u(..., i+1, ...)
            sl_curr[axis] = slice(1, -1)     # u(..., i, ...)
            sl_prev[axis] = slice(None, -2)  # u(..., i-1, ...)

            # Convert back to tuple
            u_next = u[tuple(sl_next)]
            u_curr = u[tuple(sl_curr)]
            u_prev = u[tuple(sl_prev)]

            # Accumulate the second derivative contribution
            lap[tuple(sl_curr)] += (u_next - 2 * u_curr + u_prev) / (self.domain.ds[axis] ** 2)

        return lap

    def step(self):
        """Must be implemented by child classes."""
        raise NotImplementedError("Each solver must implement its own time-stepping logic.")