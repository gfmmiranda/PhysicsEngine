import numpy as np

class Diffusion2D:

    def __init__(
            self, 
            domain,
            initial_u,
            dt = None, 
            k = 1.0, 
            boundary_type='Dirichlet'
            ):
        

        """
        Initialize a 2D wave equation solver using finite difference method.


        Parameters:
        ----------

        domain: Domain2D
            An instance of Domain2D defining the spatial domain and walls.

        initial_displacement: function
            A function of x, y defining the initial concentration/temperature.

        dt: float
            The time step size. If None, it is computed based on the CFL condition.

        c: float
            The wave speed. Default is 1.0.

        boundary_type: str
            'Dirichlet' or 'Neumann'.
        """

        self.name = 'Diffusion'
        self.domain = domain
        self.phi = initial_u
        self.k = k
        self.boundary_type = boundary_type
        self.dt = 0.99*(domain.dx ** 2) / (4*k) if dt is None or dt >= (domain.dx ** 2) / (4*k) else dt
        self.dim = 2

        if self.boundary_type == 'Neumann':
            self.domain.generate_neumann_map()

        self.initialize()

    def initialize(self):
        """Sets up the initial wave state based on the provided functions."""
        self.u_curr = self.phi(self.domain.X, self.domain.Y)

        self.apply_boundary_conditions(self.u_curr)

    def apply_boundary_conditions(self, u):
        """Applies the specified boundary conditions to the wave field."""

        if self.boundary_type == 'Dirichlet':
            u[~self.domain.mask] = 0.0

        elif self.boundary_type == 'Neumann':
            u[~self.domain.mask] = 0.0
            for (wall_idx, air_idx) in self.domain.neumann_map:
                u[wall_idx] = u[air_idx]

    def laplacian(self, u):
        """Compute the second spatial derivative (Laplacian) using finite differences."""
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.domain.dx**2 +
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / self.domain.dy**2
        )
        return lap
    
    def step(self):
        """Advance the wave solution by one time step."""
        u_next = self.u_curr + self.k * self.dt * self.laplacian(self.u_curr)

        # Apply boundary conditions
        self.apply_boundary_conditions(u_next)

        # Update states for next iteration
        self.u_curr = u_next