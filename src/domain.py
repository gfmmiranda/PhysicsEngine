import numpy as np

class BaseDomain:
    """
    Standardized parent class.
    Internally, everything is a NumPy array of shape (ndim,).
    """
    def __init__(self, length, dx):
        # 1. Standardize 'length' to a generic numpy array
        self.L = np.atleast_1d(np.array(length, dtype=float))
        self.ndim = self.L.size
        
        # 2. Standardize 'dx' (grid spacing)
        # If user passes single float, assume isotropic (same dx for x, y, z)
        dx_input = np.atleast_1d(np.array(dx, dtype=float))
        if dx_input.size == 1:
            self.ds = np.repeat(dx_input, self.ndim) # e.g. [0.1, 0.1]
        else:
            self.ds = dx_input # Anisotropic grid e.g. [0.1, 0.5]
            
        # 3. Calculate 'N' (Number of points) automatically
        # Vectorized calculation: Works for 1D, 2D, 3D instantly
        self.N = (np.floor(self.L / self.ds) + 1).astype(int)
        
        # 4. Initialize placeholders
        self.mask = None
        self.is_wall = None

    def update_boundaries(self):
        """Standard N-dimensional boundary enforcer."""
        if self.mask is None: return

        self.is_wall = ~self.mask
        
        # Dynamic slicing to force edges to be walls
        # This loop works for any number of dimensions
        for axis in range(self.ndim):
            # Create a slice object that selects "Everything"
            sl_start = [slice(None)] * self.ndim
            sl_end = [slice(None)] * self.ndim
            
            # Target the first and last index of the current axis
            sl_start[axis] = 0
            sl_end[axis] = -1
            
            self.is_wall[tuple(sl_start)] = True
            self.is_wall[tuple(sl_end)] = True
        
        self.mask = ~self.is_wall

    def generate_neumann_map(self):
        """Standard N-dimensional Neumann map generator."""
        self.neumann_map = []
        
        # np.where returns a tuple of arrays, one per dimension
        wall_coords = np.where(self.is_wall)
        # Zip into coordinate tuples: (x,y) or (x,y,z)
        wall_points = list(zip(*wall_coords))

        for point in wall_points:
            neighbor = self._find_air_neighbor(point)
            if neighbor:
                self.neumann_map.append((point, neighbor))

    def _find_air_neighbor(self, point):
        """Scans adjacent cells in all N dimensions."""
        for axis in range(self.ndim):
            for direction in [-1, 1]:
                # Copy coordinate and shift
                probe = list(point)
                probe[axis] += direction
                probe = tuple(probe)
                
                # Check Bounds
                if 0 <= probe[axis] < self.N[axis]:
                    # Check if Air
                    if self.mask[probe]:
                        return probe
        return None
    
    def physical_to_index(self, pos):
        """
        Converts physical coordinates (e.g., [2.5, 3.0]) 
        to grid indices (e.g., (25, 30)).
        """
        pos = np.atleast_1d(pos)
        indices = []
        
        for i, p in enumerate(pos):
            # Calculate index: p / dx
            idx = int(round(p / self.ds[i]))
            
            # Clamp to safe bounds [0, N-1]
            idx = max(0, min(idx, self.N[i] - 1))
            indices.append(idx)
            
        return tuple(indices)

class Domain1D(BaseDomain):
    """Specific implementation for 1D lines."""
    def __init__(self, length, dx):
        # Pass inputs up; BaseDomain converts them to arrays [L], [dx]
        super().__init__(length, dx)
        
        # Create 1D Grid
        self.x = np.linspace(0, self.L[0], self.N[0])
        self.grids = (self.x,)
        
        # Create Mask (1D array)
        self.mask = np.ones(self.N[0], dtype=bool)
        self.update_boundaries()

class Domain2D(BaseDomain):
    def __init__(self, length, dx):
        # Length can be [10, 20] or just 10 (square)
        if np.ndim(length) == 0: length = [length, length]
        super().__init__(length, dx)
        
        # Create 2D Grid
        self.x = np.linspace(0, self.L[0], self.N[0])
        self.y = np.linspace(0, self.L[1], self.N[1])
        # indexing='ij' is crucial for matrix notation (row, col) -> (x, y)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.grids = (self.X, self.Y)
        
        # Create Mask (2D array)
        self.mask = np.ones(tuple(self.N), dtype=bool)
        self.update_boundaries()

    def add_rectangular_obstacle(self, pos, size):
        """
        pos: [x, y] center or bottom-left
        size: [width, height]
        """
        # Vectorized check
        x_cond = (self.X >= pos[0]) & (self.X <= pos[0] + size[0])
        y_cond = (self.Y >= pos[1]) & (self.Y <= pos[1] + size[1])
        
        self.mask[x_cond & y_cond] = False
        self.update_boundaries()

    def add_circular_cavity(self, pos, radius):
        """
        Masks everything OUTSIDE the circle (creates a circular domain).
        """
        dist_sq = (self.X - pos[0])**2 + (self.Y - pos[1])**2
        
        # 1. Identify points OUTSIDE the circle
        is_outside = (dist_sq > radius**2)
        
        # 2. Set outside points to False (Wall)
        self.mask[is_outside] = False
        
        self.update_boundaries()