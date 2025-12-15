import numpy as np
from scipy.signal import find_peaks

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

def find_first_arrival(signal, threshold_ratio=0.1):
    """
    Finds the index of the first significant peak (Direct Sound).
    
    Args:
        signal (array): The audio signal (pressure).
        threshold_ratio (float): Sensitivity (0.1 means peak must be 10% of max).
    
    Returns:
        int: Index of the first arrival.
    """
    
    # 2. Define a height threshold relative to the global max
    #    (This filters out silence/sensor noise)
    max_val = np.max(signal)
    min_height = max_val * threshold_ratio
    
    # 3. Find all peaks that are at least this tall
    #    'distance' ensures we don't pick jittery noise on the same wave
    peaks, _ = find_peaks(signal, height=min_height, distance=5)
    
    if len(peaks) > 0:
        return peaks[0]  # <--- The first one is the Direct Sound!
    else:
        return np.argmax(signal) # Fallback if signal is too weak
