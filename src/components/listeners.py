from typing import Optional, Tuple, Union, List
import numpy as np


class Listener:
    """
    Point receiver for recording field values over time.
    
    Parameters
    ----------
    pos : float or list of float
        Physical position coordinates.
    tag : str, default='mic'
        Label for plotting and debugging.
    
    Attributes
    ----------
    pos : np.ndarray
        Listener position.
    grid_idx : tuple of int or None
        Grid index assigned by domain upon registration.
    """

    def __init__(self, pos: Union[float, List[float]], tag: str = 'mic') -> None:
        self.pos = np.atleast_1d(np.array(pos, dtype=float))
        self.tag = tag
        self.grid_idx: Optional[Tuple[int, ...]] = None
        self.history: List[float] = []
        self.times: List[float] = []

    def register(self, domain) -> None:
        """
        Calculates grid index based on the domain geometry.
        Called automatically when added to a Domain.
        """
        # 1. Calculate Grid Index
        self.grid_idx = domain.physical_to_index(self.pos)
        
        # 2. Safety Check (Wall Detection)
        # We access the mask directly using the index we just calculated
        if hasattr(domain, 'mask') and domain.mask is not None:
            if not domain.mask[self.grid_idx]:
                print(f"⚠️ Warning: Listener '{self.tag}' at {self.pos} is inside a wall!")

    def reset(self) -> None:
        """Clear recorded data for a new simulation run."""
        self.history = []
        self.times = []

    def record(self, t: float, u_field: np.ndarray) -> None:
        if self.grid_idx is None:
            raise ValueError(f"Listener '{self.tag}' has not been registered with a domain.")
        
        # Handle both Scalar (GPU/Optimized) and Array (CPU) inputs
        if np.isscalar(u_field) or isinstance(u_field, (float, int)):
            val = u_field
        else:
            val = u_field[self.grid_idx]
            
        self.history.append(val)
        self.times.append(t)

    def get_time_series(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.times), np.array(self.history)

    def compute_spectrum(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute frequency spectrum using Real FFT."""
        times, signal = self.get_time_series()
        n = len(signal)
        if n < 2: return None, None
        
        dt = times[1] - times[0]
        # Safety check for duplicate timestamps (e.g. from JAX warmup)
        if dt <= 1e-9:
            dt = np.mean(np.diff(times))
        
        fft_data = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        
        return freqs, np.abs(fft_data)