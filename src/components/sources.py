from typing import Optional, Tuple, Union, List
import numpy as np


class HarmonicSource:
    """
    Point source emitting a sinusoidal signal.
    
    Parameters
    ----------
    pos : float or list of float
        Physical position coordinates.
    frequency : float
        Oscillation frequency in Hz.
    amplitude : float, default=1.0
        Signal amplitude.
    phase : float, default=0.0
        Initial phase in radians.
    
    Attributes
    ----------
    pos : np.ndarray
        Source position.
    frequency : float
        Oscillation frequency.
    amplitude : float
        Signal amplitude.
    phase : float
        Phase offset.
    grid_idx : tuple of int or None
        Grid index assigned by domain upon registration.
    """

    def __init__(
        self,
        pos: Union[float, List[float]],
        frequency: float,
        amplitude: float = 1.0,
        phase: float = 0.0
    ) -> None:
        self.pos = np.atleast_1d(np.array(pos, dtype=float))
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.grid_idx: Optional[Tuple[int, ...]] = None

    def value(self, t: float) -> float:
        """
        Compute signal value at time t.
        
        Parameters
        ----------
        t : float
            Current simulation time.
        
        Returns
        -------
        float
            Signal amplitude: A*sin(ωt + φ).
        """
        omega = 2 * np.pi * self.frequency
        return self.amplitude * np.sin(omega * t + self.phase)


class RickerSource:
    """
    Ricker wavelet (Mexican hat) point source.
    
    Commonly used in seismic and acoustic simulations as a broadband
    pulse with well-defined spectral content centered at the peak frequency.
    
    Parameters
    ----------
    pos : float or list of float
        Physical position coordinates.
    peak_freq : float
        Central frequency of the wavelet in Hz.
    delay : float
        Time delay before wavelet peak in seconds.
    amplitude : float, default=1.0
        Signal amplitude scaling.
    
    Attributes
    ----------
    pos : np.ndarray
        Source position.
    fp : float
        Peak frequency.
    dr : float
        Time delay.
    amp : float
        Amplitude.
    grid_idx : tuple of int or None
        Grid index assigned by domain upon registration.
    """

    def __init__(
        self,
        pos: Union[float, List[float]],
        peak_freq: float,
        delay: float,
        amplitude: float = 1.0
    ) -> None:
        self.pos = np.atleast_1d(np.array(pos, dtype=float))
        self.grid_idx: Optional[Tuple[int, ...]] = None
        self.fp = peak_freq
        self.dr = delay
        self.amp = amplitude

    def value(self, t: float) -> float:
        """
        Compute Ricker wavelet value at time t.
        
        Parameters
        ----------
        t : float
            Current simulation time.
        
        Returns
        -------
        float
            Wavelet amplitude: A*(1 - 2τ²)*exp(-τ²) where τ = π*fp*(t - delay).
        """
        tau = np.pi * self.fp * (t - self.dr)
        return self.amp * (1 - 2 * tau**2) * np.exp(-tau**2)