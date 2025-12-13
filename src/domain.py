import numpy as np

class Domain2D:
    """Defines a 2D rectangular domain with walls and obstacles."""
    def __init__(self, length, dx):
        self.L = length
        self.dx = dx
        self.dy = dx
        
        # 1. Create the base grid
        self.N = [int(self.L[0] / dx) + 1, int(self.L[1] / dx) + 1]
        self.x = np.linspace(0, self.L[0], self.N[0])
        self.y = np.linspace(0, self.L[1], self.N[1])
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # 2. The Mask: True = Active Medium, False = Wall
        # Initialize as all active (empty rectangular room)
        self.mask = np.ones_like(self.X, dtype=bool)

        # 3. Boundary Indices (Optimization for the solver)
        self.update_boundaries()

    def update_boundaries(self):
        """Pre-calculates where the walls are for fast enforcement."""
        # The boundary is anywhere the mask is False OR the edge of the array
        self.is_wall = ~self.mask
        
        # Also force edges of the domain to be walls (optional, but safe)
        self.is_wall[0, :] = True
        self.is_wall[-1, :] = True
        self.is_wall[:, 0] = True
        self.is_wall[:, -1] = True
        
        # Sync mask with walls
        self.mask = ~self.is_wall

    def add_rectangular_obstacle(self, pos, length):
        """Carves a rectangle out of the active domain (makes it a wall)."""
        obstacle = (self.X >= pos[0]) & (self.X <= pos[0] + length[0]) & \
                   (self.Y >= pos[1]) & (self.Y <= pos[1] + length[1])
        self.mask[obstacle] = False
        self.update_boundaries()

    def add_circular_cavity(self, pos, radius):
        """Creates a circular wall cavity in the domain."""
        dist_sq = (self.X - pos[0])**2 + (self.Y - pos[1])**2
        cavity = (dist_sq <= radius**2)
        self.mask[~cavity] = False
        self.update_boundaries()

    def generate_neumann_map(self):
        """Generates a map for Neumann boundary conditions."""
        self.neumann_map = []
        wall_x, wall_y = np.where(self.is_wall)

        # For each wall cell, find a neighboring 'Air' cell to copy from
        for x, y in zip(wall_x, wall_y):
            # Check air neighbors
            neighbors = [
                (x+1, y), (x-1, y), (x, y+1), (x, y-1)
            ]

            valid_sources = []
            for nx, ny in neighbors:
                # Ensure neighbor is inside grid bounds
                if 0 <= nx < self.N[0] and 0 <= ny < self.N[1]:
                    if self.mask[nx, ny]: # If neighbor is Air
                        valid_sources.append((nx, ny))

            # If we found valid air neighbors, pick one (average or single)
            # For simplicity, we just pick the first one found.
            if valid_sources:
                target_y, target_x = valid_sources[0]
                self.neumann_map.append( ((x,y), (target_x, target_y)) )
