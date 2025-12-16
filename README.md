# PhysicsEngine: Finite Difference PDE Solver

A modular, Python-based physics engine for solving Partial Differential Equations (PDEs) using the Finite Difference Time Domain (FDTD) method. Currently supports acoustic wave propagation and heat diffusion in 1D and 2D domains.

## 🚀 Features

### Core Solvers
* **Wave Equation:** 2nd-order accurate FDTD solver for acoustic pressure fields.
* **Heat Equation:** Explicit finite difference solver for thermal diffusion.

### Advanced Boundary Conditions
* **Dirichlet:** Fixed value boundaries (e.g., u=0).
* **Neumann:** Reflective boundaries (slope=0).
* **Robin:** Frequency-independent absorption (impedance) boundaries for room acoustics.

### Simulation Components
* **Dynamic Sources:** Support for Harmonic oscillators and Ricker Wavelets (band-limited pulses).
* **Listeners:** Virtual sensors to record time-series data at specific grid points.
* **Geometry:** Flexible 1D and 2D domain generation with obstacle masking.

### Visualization
* **Interactive Animation:** 3D surface plots (2D domains) and line plots (1D domains) using Plotly.
* **Validation Tools:** Comparison utilities against Geometric Acoustics (Image Source Method).

---

## 📂 Project Structure

The project is organized into logical modules for scalability:

```text
src/
├── core/                   # Fundamental grids and base solver logic
│   ├── domain.py           # Finite difference grid and geometry masking
│   └── pdesolver.py        # Abstract PDE solver handling time-stepping
│
├── solvers/                # Physics engines
│   ├── wave.py             # Acoustic Wave Equation solver
│   └── heat.py             # Heat Diffusion solver
│
├── components/             # Simulation objects
│   ├── sources.py          # Ricker, Harmonic, and Point sources
│   └── listeners.py        # Virtual microphones/sensors
│
├── utils/                  # Helper functions
│   └── utils.py            # Initial condition generators (Gaussian, etc.)
│
└── visualization/          # Rendering tools
    └── animator.py         # Interactive Plotly animator

```
### Usage Examples

### 2D Acoustic Wave Simulation with Robin Boundaries
```python

import numpy as np
from src.core import Domain2D
from src.solvers import Wave
from src.components import RickerSource, Listener
from src.visualization import PhysicsAnimator

# 1. Setup Domain
room = Domain2D(length=[10.0, 10.0], dx=0.05)

# 2. Configure Solver (c=343 m/s, Robin Boundaries for Absorption)
solver = Wave(
    domain=room, 
    c=343.0, 
    boundary_type='robin',
    alpha=0.2  # Absorption coefficient
)

# 3. Add Source and Listener
src = RickerSource(pos=[5.0, 5.0], peak_freq=200, delay=0.05)
mic = Listener(pos=[7.0, 7.0])

solver.add_dynamic_source(src)
solver.add_listener(mic)

# 4. Run & Animate
animator = PhysicsAnimator(solver, total_time=0.1)
animator.run()
fig = animator.create_animation(skip_frames=5)
fig.show()
```

### 2D Heat Diffusion from a central spot.

```python

from src.solvers import Heat
from src.utils import utils

# Setup Solver
solver = Heat(
    domain=room, 
    k=1.5,  # Thermal diffusivity
    initial_u=utils.get_initial_gaussian(pos=[5.0, 5.0], sigma=1.0)
)

# Run logic is identical to Wave solver due to shared PDESolver interface
```
