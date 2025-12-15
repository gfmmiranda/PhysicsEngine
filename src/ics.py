import numpy as np

def get_normal_mode_dirichlet(n, m, Lx, Ly):
    """
    Returns a function that computes the (n, m) normal mode displacement in a rectangular room.
    """
    def displacement(x, y):
        return np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly)
    
    return displacement

def get_normal_mode_neumann(n, m, Lx, Ly):
    """
    Returns a function that computes the (n, m) normal mode displacement in a rectangular room.
    """
    def displacement(x, y):
        return np.cos(n * np.pi * x / Lx) * np.cos(m * np.pi * y / Ly)
    
    return displacement

def get_initial_gaussian(pos, sigma):
    """
    Returns a function that computes a Gaussian pulse in N dimensions.
    pos: List or array of coordinates [x0, y0, z0...]
    sigma: Width of the pulse
    """
    # Ensure pos is an array so we can index it
    center = np.atleast_1d(np.array(pos, dtype=float))

    def displacement(*coords):
        # coords comes in as a tuple: (X,) or (X, Y) or (X, Y, Z)
        
        # 1. Start with 0.0 accumulator
        dist_sq = 0.0
        
        # 2. Iterate over each dimension provided
        for i, grid in enumerate(coords):
            # Add contribution: (x - x0)^2
            dist_sq += (grid - center[i])**2
            
        # 3. Compute Gaussian
        return np.exp(-dist_sq / (2 * sigma**2))

    return displacement
