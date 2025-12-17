from typing import Callable, Optional, List, Tuple
import numpy as np

from src.core import PDESolver
from src.core.domain import BaseDomain


class Wave(PDESolver):
    """
    FDTD solver for the wave equation with mixed boundary conditions.
    
    Solves ∂²u/∂t² = c²∇²u + f using explicit central-difference (leapfrog) scheme.
    Supports Dirichlet (fixed), Neumann (hard wall), and Robin
    boundary conditions. Robin boundaries implement a mixed condition that
    interpolates between fully reflective and Mur's absorbing boundary.
    
    Parameters
    ----------
    domain : BaseDomain
        Computational domain with materials map for spatially-varying absorption.
    initial_u : callable, optional
        Initial displacement u(x,y,0). Defaults to zero field.
    initial_ut : callable, optional
        Initial velocity ∂u/∂t(x,y,0). Defaults to zero field.
    dt : float, optional
        Time step. If None or violates CFL, computed automatically.
    c : float, default=1.0
        Wave propagation speed.
    boundary_type : str, default='dirichlet'
        Boundary condition: 'dirichlet', 'neumann', or 'robin'.
        Automatically switches to 'robin' if absorption is detected.
    R : float, optional
        Global reflection coefficient (deprecated, use domain.materials).
    
    Attributes
    ----------
    c : float
        Wave speed.
    phi : callable
        Initial displacement function.
    psi : callable
        Initial velocity function.
    R_map : np.ndarray
        Spatially-varying reflection coefficient from domain.
    dt : float
        Time step satisfying CFL condition.
    u_prev : np.ndarray
        Displacement field at time t-dt.
    u_curr : np.ndarray
        Displacement field at time t.
    boundary_physics_map : list of tuple
        Pre-computed boundary data: (wall_idx, air_idx, dn, R) for each wall point.
    """

    def __init__(
        self,
        domain: BaseDomain,
        initial_u: Callable[..., np.ndarray] = lambda x, y: np.zeros_like(x),
        initial_ut: Callable[..., np.ndarray] = lambda x, y: np.zeros_like(x),
        dt: Optional[float] = None,
        c: float = 1.0,
        boundary_type: str = 'dirichlet'
    ) -> None:
        has_absorption = np.any(domain.materials < 1.0)
        
        if has_absorption and boundary_type == 'dirichlet':
            print("Auto-switching to 'robin' boundaries (Absorption detected in Domain).")
            boundary_type = 'robin'
        
        super().__init__(domain, boundary_type)
        self.name = 'Wave'
        
        self.c = c
        self.phi = initial_u
        self.psi = initial_ut
        self.R_map = self.domain.materials
        
        # CFL condition: dt <= 1 / (c * sqrt(sum(1/dx_i^2)))
        inv_sq_sum = np.sum(1.0 / self.domain.ds**2)
        cfl_limit = 1.0 / (c * np.sqrt(inv_sq_sum))
        self.dt = 0.99 * cfl_limit if dt is None or dt >= cfl_limit else dt

        self.boundary_physics_map: List[Tuple] = []
        if self.boundary_type in ['neumann', 'robin']:
            self._compile_boundary_physics()

        self.initialize_state()

    def initialize_state(self) -> None:
        """Initialize displacement and velocity fields from stored initial conditions."""
        self.u_prev = self.phi(*self.domain.grids)
        self.u_curr = self.u_prev + self.dt * self.psi(*self.domain.grids)

        self.apply_boundary_conditions(self.u_prev)
        self.apply_boundary_conditions(self.u_curr)

    def apply_boundary_conditions(self, u: np.ndarray) -> None:
        """
        Apply acoustic boundary conditions with spatially-varying absorption.
        
        Parameters
        ----------
        u : np.ndarray
            Displacement field to modify in-place.
        
        Notes
        -----
        For each boundary point, the update interpolates between:
        - Hard wall (R=1): 2nd-order Neumann using ghost point method
        - Absorbing (R=0): 1st-order Mur absorbing boundary condition
        
        The reflection coefficient R controls the mix: u = R*u_hard + (1-R)*u_absorb
        """
        u[~self.domain.mask] = 0.0

        for (wall_idx, air_idx, dn, R) in self.boundary_physics_map:
            dt = self.dt
            c = self.c
            lam = (c * dt) / dn
            
            u_wall_n   = self.u_curr[wall_idx]
            u_wall_nm1 = self.u_prev[wall_idx]
            u_air_n    = self.u_curr[air_idx]
            u_air_next = u[air_idx]

            # Mur 1st-order absorbing boundary
            coeff_absorb = (lam - 1.0) / (lam + 1.0)
            val_absorb = u_air_n + coeff_absorb * (u_air_next - u_wall_n)
            
            # 2nd-order Neumann (hard wall)
            val_hard = (2.0 * u_wall_n - u_wall_nm1 + (lam**2) * (2.0 * u_air_n - 2.0 * u_wall_n))
            
            # Linear interpolation based on reflection coefficient
            u[wall_idx] = R * val_hard + (1.0 - R) * val_absorb

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
            local_R = self.R_map[wall_idx]
            self.boundary_physics_map.append((wall_idx, air_idx, dn, local_R))

    def step(self) -> None:
        """
        Advance solution by one time step using leapfrog scheme.
        
        Computes u^{n+1} = 2u^n - u^{n-1} + (c*dt)²(∇²u + f).
        Records field values at all registered listeners.
        """
        lap = self.laplacian(self.u_curr)
        u_next = (2 * self.u_curr - self.u_prev + 
                  (self.c * self.dt)**2 * (lap + self.active_source_field()))

        self.apply_boundary_conditions(u_next)

        self.u_prev, self.u_curr = self.u_curr, u_next
        self.t += self.dt

        for listener in self.domain.listeners:
            listener.record(self.t, self.u_curr)