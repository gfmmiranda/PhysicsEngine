from typing import List, Optional
import numpy as np

from src.core.domain import BaseDomain


class PDESolver:
    """
    Base class for finite difference PDE solvers.
    
    Provides common functionality for domain binding, source field
    computation, and spatial derivative calculation. Child classes
    implement specific physics (heat, wave) and boundary conditions.
    
    Parameters
    ----------
    domain : BaseDomain
        Computational domain instance with registered sources/listeners.
    boundary_type : str, default='dirichlet'
        Boundary condition type: 'dirichlet', 'neumann', or 'robin'.
    
    Attributes
    ----------
    domain : BaseDomain
        Reference to the computational domain.
    boundary_type : str
        Active boundary condition type.
    t : float
        Current simulation time.
    """

    def __init__(
        self, 
        domain: BaseDomain, 
        boundary_type: str = 'dirichlet'
    ) -> None:
        self.domain = domain
        self.boundary_type = boundary_type
        self.t = 0.0
        
        if self.boundary_type in ['neumann', 'robin']:
            if not hasattr(self.domain, 'neumann_map'):
                self.domain.generate_neumann_map()

    def active_source_field(self) -> np.ndarray:
        """
        Compute the source field at the current time.
        
        Evaluates all sources registered in the domain and returns
        a field array with source contributions.
        
        Returns
        -------
        np.ndarray
            Source field with same shape as domain grid.
        """
        active_field = np.zeros(tuple(self.domain.N))
        for source in self.domain.sources:
            active_field[source.grid_idx] += source.value(self.t)
        return active_field
    
    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute discrete Laplacian using 2nd-order central differences.
        
        Parameters
        ----------
        u : np.ndarray
            Input scalar field.
        
        Returns
        -------
        np.ndarray
            Laplacian of the input field.
        """
        interior_slice = tuple([slice(1, -1)] * self.domain.ndim)

        lap = np.zeros_like(u)
        for axis in range(self.domain.ndim):
            sl_next = list(interior_slice)
            sl_curr = list(interior_slice)
            sl_prev = list(interior_slice)

            sl_next[axis] = slice(2, None)
            sl_curr[axis] = slice(1, -1)
            sl_prev[axis] = slice(None, -2)

            u_next = u[tuple(sl_next)]
            u_curr = u[tuple(sl_curr)]
            u_prev = u[tuple(sl_prev)]

            lap[tuple(sl_curr)] += (u_next - 2 * u_curr + u_prev) / (self.domain.ds[axis] ** 2)

        return lap
    
    def reset(self) -> None:
        """
        Reset simulation to initial state.
        
        Clears listener history and reinitializes field variables
        while preserving domain geometry and source configuration.
        """
        self.t = 0.0
        for listener in self.domain.listeners:
            if hasattr(listener, 'reset'):
                listener.reset()
        
        self.initialize_state()
        print(f"Solver reset to t=0.0s.")

    def initialize_state(self) -> None:
        """Initialize field variables. Must be implemented by child classes."""
        raise NotImplementedError("Child solver must implement initialize_state")

    def apply_boundary_conditions(self, u: np.ndarray) -> None:
        """
        Apply boundary conditions to the field.
        
        Parameters
        ----------
        u : np.ndarray
            Field array to modify in-place.
        """
        raise NotImplementedError("Child solver must implement apply_boundary_conditions")

    def step(self) -> None:
        """Advance simulation by one time step. Must be implemented by child classes."""
        raise NotImplementedError("Child solver must implement its own time-stepping logic.")