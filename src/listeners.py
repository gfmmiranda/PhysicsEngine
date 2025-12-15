import numpy as np

class Listener:
    def __init__(self, pos):
        self.pos = np.atleast_1d(np.array(pos, dtype=float))
        self.grid_idx = None  # Will be set by the solver
        self.history = []     # Raw time-domain signal
        self.times = []       # Corresponding time stamps

    def record(self, t, u_field):
        """
        Extracts value from the field at the listener's location 
        and appends to history.
        """
        if self.grid_idx is None:
            raise ValueError("Listener has not been registered with a solver (grid_idx is None).")
        
        # Extract value at the grid index
        val = u_field[self.grid_idx]
        
        self.history.append(val)
        self.times.append(t)

    def get_time_series(self):
        """Returns (Time, Amplitude) arrays for the Impulse Response."""
        return np.array(self.times), np.array(self.history)

    def compute_spectrum(self):
        """Returns (Frequency, Magnitude) arrays for the Frequency Response."""
        times, signal = self.get_time_series()
        n = len(signal)
        if n == 0: return None, None
        
        # Calculate sampling interval (dt)
        dt = times[1] - times[0] if n > 1 else 1.0
        
        # FFT (Fast Fourier Transform)
        # rfft is for Real-valued inputs (no imaginary numbers in our pressure field)
        fft_data = np.fft.rfft(signal)
        
        # Calculate the frequency bins
        freqs = np.fft.rfftfreq(n, d=dt)
        
        # Return Magnitude (absolute value of complex FFT)
        return freqs, np.abs(fft_data)