from typing import List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt


class BaseDomain:
    """
    Base class for N-dimensional computational domains.
    
    Provides the foundation for spatial discretization in FDTD simulations.
    Handles grid generation, boundary detection, coordinate transformations,
    and registration of sources/listeners.
    
    Parameters
    ----------
    length : float or list of float
        Physical dimensions of the domain in each direction.
    dx : float or list of float
        Grid spacing. If scalar, isotropic spacing is assumed.
    
    Attributes
    ----------
    L : np.ndarray
        Physical length in each dimension.
    ndim : int
        Number of spatial dimensions.
    ds : np.ndarray
        Grid spacing per dimension.
    N : np.ndarray
        Number of grid points per dimension.
    mask : np.ndarray or None
        Boolean array where True indicates interior (air) points.
    is_wall : np.ndarray or None
        Boolean array where True indicates boundary (wall) points.
    sources : list
        Registered source objects.
    listeners : list
        Registered listener objects.
    neumann_map : list of tuple
        Wall-to-air neighbor pairs for Neumann boundary conditions.
    """

    def __init__(
        self, 
        length: Union[float, List[float]], 
        dx: Union[float, List[float]]
    ) -> None:
        self.L = np.atleast_1d(np.array(length, dtype=float))
        self.ndim = self.L.size
        
        dx_input = np.atleast_1d(np.array(dx, dtype=float))
        if dx_input.size == 1:
            self.ds = np.repeat(dx_input, self.ndim)
        else:
            self.ds = dx_input
            
        self.N = (np.floor(self.L / self.ds) + 1).astype(int)
        
        self.mask: Optional[np.ndarray] = None
        self.is_wall: Optional[np.ndarray] = None
        self.sources: List = []
        self.listeners: List = []

    def add_source(self, source) -> None:
        """
        Register a source in the domain.
        
        Parameters
        ----------
        source : HarmonicSource or RickerSource
            Source object to register.
        """
        if not self._is_valid_position(source.pos):
            print(f"⚠️ Warning: Source at {source.pos} is inside a wall!")
        source.grid_idx = self.physical_to_index(source.pos)
        self.sources.append(source)

    def add_listener(self, listener) -> None:
        """
        Register a listener in the domain.
        
        Parameters
        ----------
        listener : Listener
            Listener object to register.
        """
        if not self._is_valid_position(listener.pos):
            print(f"⚠️ Warning: Listener at {listener.pos} is inside a wall!")
        listener.grid_idx = self.physical_to_index(listener.pos)
        self.listeners.append(listener)

    def _is_valid_position(self, pos: Union[float, List[float]]) -> bool:
        """Check if a position is in the air (not a wall)."""
        idx = self.physical_to_index(pos)
        return self.mask[idx]

    def update_boundaries(self) -> None:
        """
        Enforce boundary conditions at domain edges.
        
        Sets the first and last indices along each axis as wall points,
        then updates the interior mask accordingly.
        """
        if self.mask is None:
            return

        self.is_wall = ~self.mask
        
        for axis in range(self.ndim):
            sl_start = [slice(None)] * self.ndim
            sl_end = [slice(None)] * self.ndim
            
            sl_start[axis] = 0
            sl_end[axis] = -1
            
            self.is_wall[tuple(sl_start)] = True
            self.is_wall[tuple(sl_end)] = True
        
        self.mask = ~self.is_wall

    def generate_neumann_map(self) -> None:
        """
        Build wall-to-air neighbor mapping for Neumann boundary conditions.
        
        Iterates over all wall points and identifies the nearest interior
        neighbor for ghost-point extrapolation.
        """
        self.neumann_map = []
        
        wall_coords = np.where(self.is_wall)
        wall_points = list(zip(*wall_coords))

        for point in wall_points:
            neighbor = self._find_air_neighbor(point)
            if neighbor:
                self.neumann_map.append((point, neighbor))

    def _find_air_neighbor(self, point: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        """
        Find the nearest interior neighbor for a wall point.
        
        Parameters
        ----------
        point : tuple of int
            Grid indices of the wall point.
        
        Returns
        -------
        tuple of int or None
            Grid indices of the neighboring air point, or None if not found.
        """
        for axis in range(self.ndim):
            for direction in [-1, 1]:
                probe = list(point)
                probe[axis] += direction
                probe = tuple(probe)
                
                if 0 <= probe[axis] < self.N[axis]:
                    if self.mask[probe]:
                        return probe
        return None
    
    def physical_to_index(self, pos: Union[float, List[float]]) -> Tuple[int, ...]:
        """
        Convert physical coordinates to grid indices.
        
        Parameters
        ----------
        pos : float or list of float
            Physical position coordinates.
        
        Returns
        -------
        tuple of int
            Corresponding grid indices, clamped to valid bounds.
        """
        pos = np.atleast_1d(pos)
        indices = []
        
        for i, p in enumerate(pos):
            idx = int(round(p / self.ds[i]))
            idx = max(0, min(idx, self.N[i] - 1))
            indices.append(idx)
            
        return tuple(indices)

class Domain1D(BaseDomain):
    """
    One-dimensional computational domain.
    
    Parameters
    ----------
    length : float
        Physical length of the domain.
    dx : float
        Grid spacing.
    
    Attributes
    ----------
    x : np.ndarray
        1D array of grid point positions.
    grids : tuple
        Tuple containing the x coordinate array.
    """

    def __init__(self, length: float, dx: float) -> None:
        super().__init__(length, dx)
        
        self.x = np.linspace(0, self.L[0], self.N[0])
        self.grids = (self.x,)
        
        self.mask = np.ones(self.N[0], dtype=bool)
        self.update_boundaries()


class Domain2D(BaseDomain):
    """
    Two-dimensional computational domain with material support.
    
    Parameters
    ----------
    length : float or list of float
        Physical dimensions [Lx, Ly]. If scalar, creates a square domain.
    dx : float or list of float
        Grid spacing. If scalar, isotropic spacing is used.
    material : float, default=None
        Default reflection coefficient for all boundaries.
    
    Attributes
    ----------
    x : np.ndarray
        1D array of x-coordinates.
    y : np.ndarray
        1D array of y-coordinates.
    X : np.ndarray
        2D meshgrid of x-coordinates.
    Y : np.ndarray
        2D meshgrid of y-coordinates.
    grids : tuple
        Tuple containing (X, Y) meshgrids.
    materials : np.ndarray
        Reflection coefficient map for boundary absorption.
    """

    def __init__(
        self, 
        length: Union[float, List[float]], 
        dx: Union[float, List[float]], 
        material: float = None
    ) -> None:
        if np.ndim(length) == 0:
            length = [length, length]
        super().__init__(length, dx)
        
        self.x = np.linspace(0, self.L[0], self.N[0])
        self.y = np.linspace(0, self.L[1], self.N[1])
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.grids = (self.X, self.Y)
        
        self.materials = np.full(tuple(self.N), material)
        
        self.mask = np.ones(tuple(self.N), dtype=bool)
        self.update_boundaries()

    def set_material(self, mask_condition: np.ndarray, material: float) -> None:
        """
        Set reflection coefficient for a region of the domain.
        
        Parameters
        ----------
        mask_condition : np.ndarray
            Boolean mask selecting the region.
        material_value : float
            For Wave: Reflection coefficient R (0 = fully absorbing, 1 = fully reflective).
            For Heat: Convective Heat Transfer Coefficient h (0 = no advection (Neumann), 1 = full advection (Robin)).
        """
        self.materials[mask_condition] = material
    
    def add_rectangular_obstacle(
        self, 
        pos: List[float], 
        size: List[float]
    ) -> None:
        """
        Add a rectangular obstacle centered at the given position.
        
        Parameters
        ----------
        pos : list of float
            Center coordinates [x, y].
        size : list of float
            Obstacle dimensions [width, height].
        """
        x_cond = (self.X >= pos[0] - size[0]/2) & (self.X <= pos[0] + size[0]/2)
        y_cond = (self.Y >= pos[1] - size[1]/2) & (self.Y <= pos[1] + size[1]/2)
        
        self.mask[x_cond & y_cond] = False
        self.update_boundaries()

    def add_circular_cavity(self, pos: List[float], radius: float) -> None:
        """
        Create a circular domain by masking exterior points.
        
        Parameters
        ----------
        pos : list of float
            Center coordinates [x, y].
        radius : float
            Radius of the circular cavity.
        """
        dist_sq = (self.X - pos[0])**2 + (self.Y - pos[1])**2
        is_outside = (dist_sq > radius**2)
        self.mask[is_outside] = False
        self.update_boundaries()

    def add_smart_speaker(
        self, 
        center: List[float], 
        source, 
        inner_size: List[float] = [1.0, 1.0], 
        wall_width: float = 0.1
    ) -> None:
        """
        Add a U-shaped speaker enclosure with a source inside.
        
        Creates a three-walled enclosure (back, left, right) with the
        source positioned at the opening. Walls are set to fully reflective.
        
        Parameters
        ----------
        center : list of float
            Center position [x, y] of the speaker.
        source : HarmonicSource or RickerSource
            Source to place inside the enclosure.
        inner_size : list of float, default=[1.0, 1.0]
            Interior dimensions [width, height].
        wall_width : float, default=0.1
            Thickness of the enclosure walls.
        """
        cx, cy = center
        w_in, h_in = inner_size
        t = wall_width
        
        back_size = [w_in + 2*t, t]
        back_pos  = [cx, cy - h_in/2 - t/2]
        
        left_size = [t, h_in]
        left_pos  = [cx - w_in/2 - t/2, cy]
        
        right_size = [t, h_in]
        right_pos  = [cx + w_in/2 + t/2, cy]
        
        self.add_rectangular_obstacle(back_pos, back_size)
        self.add_rectangular_obstacle(left_pos, left_size)
        self.add_rectangular_obstacle(right_pos, right_size)
        
        X, Y = self.grids
        
        def get_mask(p, s):
            return (
                (X >= p[0] - s[0]/2) & (X <= p[0] + s[0]/2) & 
                (Y >= p[1] - s[1]/2) & (Y <= p[1] + s[1]/2)
            )

        speaker_mask = (get_mask(back_pos, back_size) | 
                        get_mask(left_pos, left_size) | 
                        get_mask(right_pos, right_size))
        
        self.materials[speaker_mask] = 1.0
        
        self.add_source(source)
        print(f"✅ Smart Speaker added at {center}")

    def preview(self) -> None:
        """
        Visualize the domain setup.
        
        Displays geometry, material properties, sources, and listeners
        in a single plot for debugging and verification.
        """
        plt.figure(figsize=(8, 6))
        
        visual_map = self.materials.copy()
        visual_map[self.mask] = np.nan
        
        plt.imshow(visual_map.T, origin='lower', cmap='inferno', 
                   extent=[0, self.L[0], 0, self.L[1]], vmin=0, vmax=1)
        plt.colorbar(label="Wall Reflection Coeff (R)")
        
        plt.contour(self.X, self.Y, self.mask, levels=[0.5], colors='black', linewidths=1)

        for i, s in enumerate(self.sources):
            if i == 0:
                plt.plot(s.pos[0], s.pos[1], 'r*', markersize=12, label='Source')
            else:
                plt.plot(s.pos[0], s.pos[1], 'r*', markersize=12)
            
        for j, l in enumerate(self.listeners):
            if j == 0:
                plt.plot(l.pos[0], l.pos[1], 'go', markersize=8, label='Mic')
            else:
                plt.plot(l.pos[0], l.pos[1], 'go', markersize=8)

        plt.title("Domain Preview: Geometry & Setup")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.show()