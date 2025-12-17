import numpy as np
from src.components import HarmonicSource

class PDESolver:
    """
    Base class for Finite Difference solvers.
    Acts purely as a compute engine over the Domain.
    """
    def __init__(self, domain, boundary_type='dirichlet'):
        self.domain = domain
        self.boundary_type = boundary_type

        # Static background sources
        self.source_field = np.zeros(tuple(self.domain.N))
        self.t = 0.0
        
        # Ensure Geometry is ready
        if self.boundary_type in ['neumann', 'robin']:
            if not hasattr(self.domain, 'neumann_map'):
                self.domain.generate_neumann_map()

    def active_source_field(self):
        """Computes the active source field from Domain objects."""
        active_field = self.source_field.copy()
        # Read directly from the Domain
        for source in self.domain.sources:
            # grid_idx was calculated when source was added to domain
            active_field[source.grid_idx] += source.value(self.t)

        return active_field
    
    def laplacian(self, u):
        """Standard finite difference Laplacian."""

        interior_slice = tuple([slice(1,-1)]*self.domain.ndim)

        lap = np.zeros_like(u)
        for axis in range(self.domain.ndim):
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
    
    def reset(self):
        self.t = 0.0
        # Reset listeners in the domain
        for listener in self.domain.listeners:
            if hasattr(listener, 'reset'):
                listener.reset()
        
        self.initialize_state()
        print(f"Solver reset to t=0.0s.")

    def apply_boundary_conditions(self, u):
        """Forces child classes to define their own physics."""
        raise NotImplementedError("Child solver must implement apply_boundary_conditions")

    def step(self):
        """Must be implemented by child classes."""
        raise NotImplementedError("Each solver must implement its own time-stepping logic.")