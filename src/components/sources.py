from typing import Optional, List, Tuple, Union
import numpy as np

class Source:
    """
    Abstract base class for all wave sources.
    
    Attributes
    ----------
    pos : np.ndarray
        Physical position.
    injection_points : list of (tuple, float)
        List of (grid_index, weight) tuples calculated upon registration.
        Example: [((50, 50), 1.0)] for a point source.
    """
    def __init__(self, pos: Union[float, List[float]]):
        self.pos = np.atleast_1d(np.array(pos, dtype=float))
        self.injection_points: List[Tuple[Tuple[int, ...], float]] = []

    def register(self, domain) -> None:
        """
        Calculates grid indices based on the domain geometry.
        Default implementation registers a single point source.
        """
        idx = domain.physical_to_index(self.pos)
        
        # specific check using domain methods
        if hasattr(domain, 'mask') and not domain.mask[idx]:
             print(f"⚠️ Warning: Source at {self.pos} is inside a wall!")
             
        # Standard Monopole: One point, weight 1.0
        self.injection_points = [(idx, 1.0)]

    def value(self, t: float) -> float:
        """Return the raw signal amplitude at time t."""
        raise NotImplementedError


class HarmonicSource(Source):
    def __init__(
        self,
        pos: Union[float, List[float]],
        frequency: float,
        amplitude: float = 1.0,
        phase: float = 0.0
    ) -> None:
        super().__init__(pos)
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def value(self, t: float) -> float:
        omega = 2 * np.pi * self.frequency
        return self.amplitude * np.sin(omega * t + self.phase)


class RickerSource(Source):
    def __init__(
        self,
        pos: Union[float, List[float]],
        frequency: float,
        delay: float,
        amplitude: float = 1.0
    ) -> None:
        super().__init__(pos)
        self.fp = frequency
        self.dr = delay
        self.amp = amplitude

    def value(self, t: float) -> float:
        tau = np.pi * self.fp * (t - self.dr)
        return self.amp * (1 - 2 * tau**2) * np.exp(-tau**2)


class DipoleSource(HarmonicSource):
    """
    Two anti-phase point sources separated by a small distance.
    """
    def __init__(
        self, 
        pos: Union[float, List[float]], 
        frequency: float,
        amplitude: float = 1.0,
        orientation: float = 0.0,
        separation: float = 0.1, 
        **kwargs
    ) -> None:
        super().__init__(pos, frequency, amplitude)
        self.theta = orientation
        self.sep = separation

    def register(self, domain) -> None:
        """Overrides registration to create two injection points."""
        cx, cy = self.pos[0], self.pos[1]
        
        # Calculate offsets
        dx = (self.sep / 2.0) * np.cos(self.theta)
        dy = (self.sep / 2.0) * np.sin(self.theta)

        print(dx, dy)
        
        idx_plus = domain.physical_to_index([cx + dx, cy + dy])
        idx_minus = domain.physical_to_index([cx - dx, cy - dy])
        
        # Register both points with opposite weights
        self.injection_points = [
            (idx_plus, 1.0),
            (idx_minus, -1.0)
        ]