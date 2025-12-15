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
    
class RickerSource:
    def __init__(self, pos, peak_freq, delay, amplitude=1.0):
        self.pos = np.atleast_1d(np.array(pos, dtype=float))
        self.grid_idx = None # Set by solver
        self.fp = peak_freq
        self.dr = delay
        self.amp = amplitude

    def value(self, t):
        # The "Mexican Hat" wavelet
        tau = np.pi * self.fp * (t - self.dr)
        return self.amp * (1 - 2 * tau**2) * np.exp(-tau**2)