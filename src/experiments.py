import numpy as np
import matplotlib.pyplot as plt
from src.components import Listener

class BaseExperiment:
    """
    Manages the simulation lifecycle: Setup -> Run -> Analyze.
    """
    def __init__(self, domain, solver_class, **solver_kwargs):
        self.domain = domain
        self.SolverClass = solver_class
        self.solver_kwargs = solver_kwargs
        self.solver = None
        self.results = {}

    def setup(self):
        """Place measurement microphones (Listeners). Override in subclasses."""
        pass

    def run(self, duration):
        """Execute the physics engine."""
        print(f"🧪 Starting Experiment: {self.__class__.__name__}")
        self.solver = self.SolverClass(self.domain, **self.solver_kwargs)
        
        steps = int(duration / self.solver.dt)
        print(f"   Simulating {duration:.3f}s ({steps} steps)...")
        
        for i in range(steps):
            self.solver.step()
            
        print("✅ Experiment Complete.")

    def analyze(self):
        """Process listener data. Override in subclasses."""
        raise NotImplementedError

class DirectivityExperiment(BaseExperiment):
    """
    Virtual Anechoic Chamber: Measures polar patterns.
    """
    def __init__(self, domain, solver_class, center=None, radius=2.0, num_points=72, **solver_kwargs):
        super().__init__(domain, solver_class, **solver_kwargs)
        self.center = domain.L/2.0
        self.radius = radius
        self.num_points = num_points
        self.listeners = []
        self.angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    def setup(self):
        """Deploys a ring of listeners around the center."""
        cx, cy = self.center
        self.listeners = []
        
        for theta in self.angles:
            lx = cx + self.radius * np.cos(theta)
            ly = cy + self.radius * np.sin(theta)
            
            l = Listener(pos=[lx, ly], tag=f"deg_{int(np.degrees(theta))}")
            self.domain.add_listener(l)
            self.listeners.append(l)
            
        print(f"   Deployed {len(self.listeners)} microphones at r={self.radius}m.")

    def analyze(self):
        """Extracts peak amplitude from each microphone."""
        magnitudes = []
        for l in self.listeners:
            _, signal = l.get_time_series()
            # Get steady-state peak (last 20% of signal)
            steady_state = signal[int(len(signal)*0.8):]
            if len(steady_state) == 0: steady_state = signal # Fallback for short runs
            magnitudes.append(np.max(np.abs(steady_state)))
            
        self.results['angles'] = self.angles
        self.results['magnitudes'] = np.array(magnitudes)
        
        return self.results

    def plot(self):
        """Generates the Polar Plot."""
        if 'magnitudes' not in self.results:
            self.analyze()
            
        theta = np.concatenate([self.results['angles'], [self.results['angles'][0]]])
        r = np.concatenate([self.results['magnitudes'], [self.results['magnitudes'][0]]])
        
        # Normalize
        r_norm = r / (np.max(r) + 1e-12)

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta, r_norm, linewidth=2, label='Measured')
        ax.fill(theta, r_norm, alpha=0.3)
        
        ax.set_theta_zero_location("E")
        ax.set_title("Directivity Pattern", va='bottom')
        plt.legend()
        plt.show()


class ImpulseResponseExperiment(BaseExperiment):
    """
    Measures the System Response (Impulse Response & Frequency Response).
    
    Requirements:
    - The Domain should contain a broadband source (e.g., RickerSource).
    """
    def __init__(self, domain, solver_class, measure_pos, **solver_kwargs):
        super().__init__(domain, solver_class, **solver_kwargs)
        self.measure_pos = measure_pos
        self.listener = None
        
    def setup(self):
        """Places a measurement microphone."""
        self.listener = Listener(pos=self.measure_pos, tag="measurement_mic")
        self.domain.add_listener(self.listener)
        print(f"   Microphone placed at {self.measure_pos}")

    def analyze(self):
        """Computes FFT of the recorded signal."""
        if not self.listener:
            raise ValueError("Experiment not set up.")
            
        times, signal = self.listener.get_time_series()
        freqs, mag = self.listener.compute_spectrum()
        
        # Convert to dB (avoid log(0) errors)
        mag_db = 20 * np.log10(mag + 1e-12)
        
        # Normalize peak to 0 dB for readability
        mag_db = mag_db - np.max(mag_db)
        
        self.results['times'] = times
        self.results['signal'] = signal
        self.results['freqs'] = freqs
        self.results['mag_db'] = mag_db
        
        return self.results

    def plot(self):
        """Plots Time Domain (IR) and Frequency Domain (FR)."""
        if 'signal' not in self.results:
            self.analyze()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Impulse Response (Time)
        ax1.plot(self.results['times'], self.results['signal'], color='black', lw=1)
        ax1.set_title("Impulse Response (Time Domain)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        # 2. Frequency Response (Bode Magnitude)
        ax2.semilogx(self.results['freqs'], self.results['mag_db'], color='tab:blue', lw=1.5)
        ax2.set_title("Frequency Response (0dB Normalized)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_xlim(20, 20000) # Audio range
        ax2.set_ylim(-60, 5)    # Typical dynamic range
        ax2.grid(True, which="both", alpha=0.3)
        
        plt.tight_layout()
        plt.show()