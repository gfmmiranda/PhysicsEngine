from typing import Callable
import numpy as np
from scipy.signal import find_peaks


def get_normal_mode_dirichlet(
    n: int,
    m: int,
    Lx: float,
    Ly: float
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Generate Dirichlet normal mode shape for a rectangular domain.
    
    Parameters
    ----------
    n : int
        Mode number in x-direction.
    m : int
        Mode number in y-direction.
    Lx : float
        Domain length in x.
    Ly : float
        Domain length in y.
    
    Returns
    -------
    callable
        Function computing sin(nπx/Lx) * sin(mπy/Ly).
    """
    def displacement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly)
    
    return displacement


def get_normal_mode_neumann(
    n: int,
    m: int,
    Lx: float,
    Ly: float
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Generate Neumann normal mode shape for a rectangular domain.
    
    Parameters
    ----------
    n : int
        Mode number in x-direction.
    m : int
        Mode number in y-direction.
    Lx : float
        Domain length in x.
    Ly : float
        Domain length in y.
    
    Returns
    -------
    callable
        Function computing cos(nπx/Lx) * cos(mπy/Ly).
    """
    def displacement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.cos(n * np.pi * x / Lx) * np.cos(m * np.pi * y / Ly)
    
    return displacement


def get_initial_gaussian(
    pos: list,
    sigma: float
) -> Callable[..., np.ndarray]:
    """
    Generate an N-dimensional Gaussian pulse function.
    
    Parameters
    ----------
    pos : list of float
        Center coordinates [x0, y0, ...].
    sigma : float
        Standard deviation (pulse width).
    
    Returns
    -------
    callable
        Function computing exp(-|r - r0|² / 2σ²) for any dimension.
    """
    center = np.atleast_1d(np.array(pos, dtype=float))

    def displacement(*coords: np.ndarray) -> np.ndarray:
        dist_sq = 0.0
        for i, grid in enumerate(coords):
            dist_sq += (grid - center[i])**2
        return np.exp(-dist_sq / (2 * sigma**2))

    return displacement


def find_first_arrival(signal: np.ndarray, threshold_ratio: float = 0.1) -> int:
    """
    Detect the first significant peak in a signal (direct sound arrival).
    
    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal.
    threshold_ratio : float, default=0.1
        Minimum peak height as fraction of global maximum.
    
    Returns
    -------
    int
        Index of first arrival. Falls back to global maximum if no peaks found.
    """
    max_val = np.max(signal)
    min_height = max_val * threshold_ratio
    
    peaks, _ = find_peaks(signal, height=min_height, distance=5)
    
    if len(peaks) > 0:
        return peaks[0]
    else:
        return np.argmax(signal)

