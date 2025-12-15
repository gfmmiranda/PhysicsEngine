import numpy as np

class HarmonicSource:
    """
    Represents a point source emitting a harmonic wave.
    """
    def __init__(self, pos, frequency, amplitude=1.0, phase=0.0):
        # Store as array for safety
        self.pos = np.atleast_1d(np.array(pos, dtype=float)) 
        
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        
        # Calculated by the Solver
        self.grid_idx = None 

    def value(self, t):
        """Returns the signal strength at time t."""
        omega = 2 * np.pi * self.frequency
        return self.amplitude * np.sin(omega * t + self.phase)