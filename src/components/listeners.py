from typing import Optional, Tuple, Union, List
import numpy as np


class Listener:
    """
    Point receiver for recording field values over time.
    
    Records the time series at a fixed spatial location, enabling
    impulse response and frequency response analysis.
    
    Parameters
    ----------
    pos : float or list of float
        Physical position coordinates.
    
    Attributes
    ----------
    pos : np.ndarray
        Listener position.
    grid_idx : tuple of int or None
        Grid index assigned by domain upon registration.
    history : list of float
        Recorded field values.
    times : list of float
        Corresponding time stamps.
    """

    def __init__(self, pos: Union[float, List[float]]) -> None:
        self.pos = np.atleast_1d(np.array(pos, dtype=float))
        self.grid_idx: Optional[Tuple[int, ...]] = None
        self.history: List[float] = []
        self.times: List[float] = []

    def reset(self) -> None:
        """Clear recorded data for a new simulation run."""
        self.history = []
        self.times = []

    def record(self, t: float, u_field: np.ndarray) -> None:
        """
        Sample the field at the listener location.
        
        Parameters
        ----------
        t : float
            Current simulation time.
        u_field : np.ndarray
            Current field state.
        
        Raises
        ------
        ValueError
            If listener has not been registered with a domain.
        """
        if self.grid_idx is None:
            raise ValueError("Listener has not been registered with a solver.")
        
        if np.isscalar(u_field) or isinstance(u_field, (float, int)):
            val = u_field
        else:
            val = u_field[self.grid_idx]
            
        self.history.append(val)
        self.times.append(t)

    def get_time_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the recorded time series.
        
        Returns
        -------
        times : np.ndarray
            Time stamps array.
        signal : np.ndarray
            Recorded amplitude values.
        """
        return np.array(self.times), np.array(self.history)

    def compute_spectrum(
        self
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute the frequency spectrum of the recorded signal via FFT.
        
        Returns
        -------
        freqs : np.ndarray or None
            Frequency bins in Hz.
        magnitude : np.ndarray or None
            Magnitude spectrum. Returns (None, None) if insufficient data.
        """
        times, signal = self.get_time_series()
        n = len(signal)
        if n < 2:
            return None, None
        
        dt = times[1] - times[0]
        
        if dt <= 1e-9:
            print("⚠️ Warning: Duplicate timestamps detected (dt=0). Spectrum may be invalid.")
            dt = np.mean(np.diff(times))
        
        fft_data = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        
        return freqs, np.abs(fft_data)