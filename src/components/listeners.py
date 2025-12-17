import numpy as np

class Listener:
    def __init__(self, pos):
        self.pos = np.atleast_1d(np.array(pos, dtype=float))
        self.grid_idx = None
        self.history = []
        self.times = []

    def reset(self):
        """Clears recorded history to prepare for a new run."""
        # FIX: Ensure these match the variable names in __init__
        self.history = []
        self.times = []

    def record(self, t, u_field):
        if self.grid_idx is None:
             raise ValueError("Listener has not been registered with a solver.")
        
        if np.isscalar(u_field) or isinstance(u_field, (float, int)):
            val = u_field
        else:
            val = u_field[self.grid_idx]
            
        self.history.append(val)
        self.times.append(t)

    def get_time_series(self):
        return np.array(self.times), np.array(self.history)

    def compute_spectrum(self):
        times, signal = self.get_time_series()
        n = len(signal)
        if n < 2: return None, None
        
        # Robust dt calculation
        dt = times[1] - times[0]
        
        # Safety check for the "Divide by Zero" bug
        if dt <= 1e-9:
            print("⚠️ Warning: Duplicate timestamps detected (dt=0). Spectrum may be invalid.")
            # Fallback: try to find the average step from the rest of the array
            dt = np.mean(np.diff(times))
        
        fft_data = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        
        return freqs, np.abs(fft_data)