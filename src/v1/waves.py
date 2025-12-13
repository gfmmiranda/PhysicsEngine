import numpy as np

class Wave1D:

    def __init__(
            self,
            initial_displacement,
            initial_velocity,
            boundary_conditions,
            length,
            num_points = None,
            dx = None,
            dt = None,
            wave_speed = 1.0):
        
        """
        Initialize a 1D wave equation solver using finite difference method.


        Parameters:
        ----------

        initial_displacement: function
            A function of x defining the initial displacement of the wave.

        initial_velocity: function
            A function of x defining the initial velocity of the wave.

        boundary_conditions: tuple
            A tuple defining the boundary conditions at both ends of the domain.
            Each element can be 'Dirichlet' or 'Neumann'.

        length: float
            The length of the 1D domain.

        num_points: int
            The number of spatial discretization points. If None, it is computed as length/dx + 1.
            Either num_points or dx must be provided.

        dx: float
            The spatial step size. If None, it is computed as length / (num_points - 1).
            Either num_points or dx must be provided.

        dt: float
            The time step size. If None, it is computed based on the CFL condition.

        c: float
            The wave speed. Default is 1.0.
        """

        # Initialize parameters
        self.phi = initial_displacement
        self.psi = initial_velocity
        self.boundary = boundary_conditions
        self.L = length
        self.c = wave_speed
        self.dim = 1

        # Determine spatial discretization
        if num_points is None and dx is None:
            raise ValueError("Either num_points or dx must be provided.")
        
        elif num_points is not None:
            self.dx = length / (num_points - 1)
            self.N = num_points

        else:
            self.dx = dx
            self.N = int(length / dx) + 1

        # Determine time step based on CFL condition if not provided
        if dt is None:
            self.dt = self.dx / self.c * 0.99  # CFL condition

        else:
            if dt > self.dx / self.c:
                raise ValueError("Time step must be smaller than dx/c for stability.")
            self.dt = dt

        # Initialize grid with initial and Boundary Conditions
        self.initialize_grid()
            
    def initialize_grid(self):
        """Initialize the spatial grid and set initial conditions."""
        self.x = np.linspace(0, self.L, self.N)
        self.u_prev = self.phi(self.x)
        self.u_curr = self.u_prev + self.dt * self.psi(self.x)

        # Apply boundary conditions to initial state
        self.apply_boundary_conditions(self.u_prev)
        self.apply_boundary_conditions(self.u_curr)

    def apply_boundary_conditions(self, u):
        """Apply boundary conditions to the wave state array u."""

        if self.boundary[0] == 'Dirichlet':
            u[0] = 0.0
        elif self.boundary[0] == 'Neumann':
            u[0] = u[1]

        if self.boundary[1] == 'Dirichlet':
            u[-1] = 0.0
        elif self.boundary[1] == 'Neumann':
            u[-1] = u[-2]

    def laplacian(self, u):
        """Compute the second spatial derivative (Laplacian) using finite differences."""
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / self.dx**2
        return lap
    
    def step(self):
        """Advance the wave solution by one time step."""
        u_next = (2 * self.u_curr - self.u_prev +
                  (self.c * self.dt)**2 * self.laplacian(self.u_curr))

        # Apply boundary conditions
        self.apply_boundary_conditions(u_next)

        # Update states for next iteration
        self.u_prev, self.u_curr = self.u_curr, u_next

class Wave2D:

    def __init__(
            self,
            initial_displacement,
            initial_velocity,
            boundary_conditions,
            length,
            num_points = None,
            dx = None,
            dt = None,
            wave_speed = 1.0
            ):
        
        """
        Initialize a 1D wave equation solver using finite difference method.


        Parameters:
        ----------

        initial_displacement: function
            A function of x defining the initial displacement of the wave.

        initial_velocity: function
            A function of x defining the initial velocity of the wave.

        boundary_conditions: list of lists
            The domain has 2 edges for each dimension. Syntax is [[bc_x0, bc_x1], [bc_y0, bc_y1]].
            Each element can be 'Dirichlet' or 'Neumann'.

        length: list
            The length [x, y] of the 2D domain.

        num_points: int
            The number of spatial discretization points. If None, it is computed as length/dx + 1.
            Either num_points or dx must be provided.

        dx: float
            The spatial step size along dimension x. If None, it is computed as length / (num_points - 1).
            Either num_points or dx must be provided. The 2D Grid is assumed square.

        dt: float
            The time step size. If None, it is computed based on the CFL condition.

        c: float
            The wave speed. Default is 1.0.
        """

        # Initialize parameters
        self.phi = initial_displacement
        self.psi = initial_velocity
        self.boundary = boundary_conditions
        self.L = length
        self.c = wave_speed
        self.dim = 2

        # Determine spatial discretization
        if num_points is None and dx is None:
            raise ValueError("Either num_points or dx must be provided.")
        
        elif num_points is not None:
            self.dx = length[0] / (num_points[0] - 1)
            self.dy = length[1] / (num_points[1] - 1)
            self.N = num_points

        else:
            self.dx = dx
            self.dy = dx
            self.N = int(length / dx) + 1

        # Determine time step based on CFL condition if not provided
        if dt is None:
            self.dt = (self.dx / self.c) / (2**0.5)  # CFL condition in 2D

        else:
            if dt > (self.dx / self.c ) / (2**0.5):
                raise ValueError("Time step must be smaller than dx/c for stability.")
            self.dt = dt

        # Initialize grid with initial and Boundary Conditions
        self.initialize_grid()

    def initialize_grid(self):
        """Initialize the spatial grid and set initial conditions."""
        self.x = np.linspace(0, self.L[0], self.N[0])
        self.y = np.linspace(0, self.L[1], self.N[1])
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.u_prev = self.phi(self.X, self.Y)
        self.u_curr = self.u_prev + self.dt * self.psi(self.X, self.Y)

        # Apply boundary conditions to initial state
        self.apply_boundary_conditions(self.u_prev)
        self.apply_boundary_conditions(self.u_curr)

    def apply_boundary_conditions(self, u):
        """Apply boundary conditions to the wave state array u."""

        if self.boundary[0][0] == 'Dirichlet':
            u[0, :] = 0.0
        elif self.boundary[0][0] == 'Neumann':
            u[0, :] = u[1, :]

        if self.boundary[0][1] == 'Dirichlet':
            u[-1, :] = 0.0
        elif self.boundary[0][1] == 'Neumann':
            u[-1, :] = u[-2, :]

        if self.boundary[1][0] == 'Dirichlet':
            u[:, 0] = 0.0
        elif self.boundary[1][0] == 'Neumann':
            u[:, 0] = u[:, 1]

        if self.boundary[1][1] == 'Dirichlet':
            u[:, -1] = 0.0
        elif self.boundary[1][1] == 'Neumann':
            u[:, -1] = u[:, -2]

    def laplacian(self, u):
        """Compute the second spatial derivative (Laplacian) using finite differences."""
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.dx**2 +
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dy**2
        )
        return lap
    
    def step(self):
        """Advance the wave solution by one time step."""
        u_next = (2 * self.u_curr - self.u_prev +
                  (self.c * self.dt)**2 * self.laplacian(self.u_curr))

        # Apply boundary conditions
        self.apply_boundary_conditions(u_next)

        # Update states for next iteration
        self.u_prev, self.u_curr = self.u_curr, u_next


def get_normal_mode(n, m, Lx, Ly):
    """
    Returns a function that computes the (n, m) normal mode displacement.
    """
    def displacement(x, y):
        return np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly)
    
    return displacement