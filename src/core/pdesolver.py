import numpy as np
from src.components import HarmonicSource

class PDESolver:
    """
    Base class for 1, 2 or 3D finite difference solvers. 
    Handles domain binding, boundary conditions, and spatial derivatives.
    """
    def __init__(self, domain, boundary_type='dirichlet'):
        self.domain = domain
        self.boundary_type = boundary_type

        # Static background sources (e.g. constant heat)
        self.source_field = np.zeros(tuple(self.domain.N))

        # Dynamic sources (e.g. harmoni oscillators)
        self.dynamic_sources = []

        # List for possible listeners
        self.listeners = []

        # Simulation time
        self.t = 0.0
        
        # Ensure Geometry is ready
        if self.boundary_type in ['neumann', 'robin']:
            if not hasattr(self.domain, 'neumann_map'):
                self.domain.generate_neumann_map()

    def add_source_field(self, pos, value):
        """
        Generic method to add a point source.
        """
        idx = self.domain.physical_to_index(pos)
        self.source_field[idx] += value

    def add_dynamic_source(self, source: HarmonicSource):
        """
        Adds a dynamic source to the solver.
        The source's grid index is computed and stored.
        """
        source.grid_idx = self.domain.physical_to_index(source.pos)
        self.dynamic_sources.append(source)

    def active_source_field(self):
        """Computes the active source field at the current time."""
        active_field = self.source_field.copy()
        for source in self.dynamic_sources:
            active_field[source.grid_idx] += source.value(self.t)
        return active_field
    
    def add_listener(self, listener):
        """
        Registers a listener object.
        Calculates grid index once for efficiency.
        """
        listener.grid_idx = self.domain.physical_to_index(listener.pos)
        self.listeners.append(listener)

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
        """
        Resets the simulation to t=0, clears listeners, and restores initial conditions.
        Preserves: Domain, Sources, Listeners, and Boundary physics.
        """
        self.t = 0.0
        
        # 1. Clear Listener History
        for listener in self.listeners:
            if hasattr(listener, 'reset'):
                listener.reset()
        
        # 2. Re-calculate Initial Fields (Polymorphic call to child class)
        # This uses the saved self.phi / self.psi functions in Wave/Heat
        self.initialize_state()
        
        print(f"Solver reset to t=0.0s. Ready to run.")

    def apply_boundary_conditions(self, u):
        """Forces child classes to define their own physics."""
        raise NotImplementedError("Child solver must implement apply_boundary_conditions")

    def step(self):
        """Must be implemented by child classes."""
        raise NotImplementedError("Each solver must implement its own time-stepping logic.")