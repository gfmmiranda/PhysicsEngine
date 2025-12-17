from typing import Callable, Optional, List, Tuple
import numpy as np

from src.core import PDESolver
from src.core.domain import BaseDomain


class Heat(PDESolver):
    """
    FDTD solver for the heat (diffusion) equation.
    
    Solves ∂u/∂t = k∇²u + f using explicit forward-time central-space (FTCS) scheme.
    Supports Dirichlet (fixed temperature) and Neumann (insulated) boundary conditions.
    
    Parameters
    ----------
    domain : BaseDomain
        Computational domain.
    initial_u : callable
        Initial temperature distribution u(x) or u(x,y).
        Must accept grid arrays and return the initial field.
    dt : float, optional
        Time step. If None or unstable, computed from stability limit.
    k : float, default=1.0
        Thermal diffusivity coefficient.
    boundary_type : str, default='dirichlet'
        Boundary condition type: 'dirichlet', 'neumann' or robin.
    
    Attributes
    ----------
    k : float
        Thermal diffusivity.
    phi : callable
        Initial condition function (stored for reset).
    dt : float
        Time step size.
    u_curr : np.ndarray
        Current temperature field.
    """

    def __init__(
        self,
        domain: BaseDomain,
        initial_u: Callable[..., np.ndarray] = lambda *args: np.zeros_like(args[0]),
        dt: Optional[float] = None,
        k: float = 1.0,
        T_ambient: float = 0.0,
        boundary_type: str = 'robin'
    ) -> None:
        

        super().__init__(domain, boundary_type)
        self.name = 'Heat'
        
        self.k = k
        self.T_ambient = T_ambient
        self.phi = initial_u
        self.h = self.domain.materials
        
        # Stability condition: dt <= 1 / (2k * sum(1/dx_i^2))
        inv_sq_sum = np.sum(1.0 / self.domain.ds**2)
        stability_limit = 1.0 / (2 * k * inv_sq_sum)
        self.dt = 0.99 * stability_limit if dt is None or dt >= stability_limit else dt
        
        self.boundary_physics_map: List[Tuple] = []
        if self.boundary_type in ['neumann', 'robin']:
            self._compile_boundary_physics()

        self.initialize_state()
    
    def initialize_state(self) -> None:
        """Set initial temperature distribution from the stored function."""
        self.u_curr = self.phi(*self.domain.grids)
        self.apply_boundary_conditions(self.u_curr)

    
    def apply_boundary_conditions(self, u: np.ndarray) -> None:
        """
        Apply thermal boundary conditions.
        
        Parameters
        ----------
        u : np.ndarray
            Temperature field to modify in-place.
        
        Notes
        -----
        Dirichlet: Fixed temperature u=0 at walls.
        Neumann: Zero flux (∂u/∂n=0) via ghost point method.
        """
        u[~self.domain.mask] = 0.0

        # Robin boundary condition (convective heat transfer) via Ghost Point Method
        # Neumann condition is he a special case with Bi=0
        for (wall_idx, air_idx, dn, h) in self.boundary_physics_map:
            
            coeff = (2*self.k*self.dt)/(dn**2)
            local_Bi = h * dn / self.k

            u_wall_n = self.u_curr[wall_idx]
            u_air_n = self.u_curr[air_idx]

            u[wall_idx] = u_wall_n + \
                coeff * (u_air_n - u_wall_n - local_Bi*(u_wall_n - self.T_ambient))

    def _compile_boundary_physics(self) -> None:
        """
        Pre-compute boundary data for efficient time-stepping.
        
        Stores wall index, air neighbor index, normal spacing, and local
        reflection coefficient for each boundary point.
        """
        for (wall_idx, air_idx) in self.domain.neumann_map:
            axis = 0
            for dim in range(self.domain.ndim):
                if wall_idx[dim] != air_idx[dim]:
                    axis = dim
                    break
            
            dn = self.domain.ds[axis]
            local_h = self.h[wall_idx]
            self.boundary_physics_map.append((wall_idx, air_idx, dn, local_h))


    def step(self) -> None:
        """
        Advance solution by one time step using FTCS scheme.
        
        Computes u^{n+1} = u^n + dt*(k*∇²u + f).
        """
        lap = self.laplacian(self.u_curr)
        u_next = self.u_curr + self.dt * (self.k * lap + self.active_source_field())

        self.apply_boundary_conditions(u_next)
        self.u_curr = u_next
        self.t += self.dt

        for listener in self.domain.listeners:
            listener.record(self.t, self.u_curr)