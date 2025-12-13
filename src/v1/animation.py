import numpy as np
import plotly.graph_objects as go

class WaveAnimator:
    def __init__(self, solver, total_time):
        """
        Initializes the animator for both 1D and 2D solvers.

        Parameters:
        ----------
            solver: An instance of the WaveSolver class (must have .dim, .L, .N attributes)
            total_time: int or float, simulation duration
        """
        self.solver = solver
        self.total_time = total_time
        
        # Storage
        self.history = []
        self.time_steps = []
        
        # --- 1. Detect Dimension and Setup Axes ---
        # The solver logic we built uses lists for L and N (e.g., L=[1.0, 1.0])
        if self.solver.dim == 1:
            self.x_axis = np.linspace(0, self.solver.L, self.solver.N)
            self.y_axis = None
        
        elif self.solver.dim == 2:
            self.x_axis = np.linspace(0, self.solver.domain.L[0], self.solver.domain.N[0])
            self.y_axis = np.linspace(0, self.solver.domain.L[1], self.solver.domain.N[1])

    def run(self):
        """Runs the physics loop and stores data."""
        t = 0.0
        steps = int(self.total_time / self.solver.dt)
        
        print(f"Simulating {self.total_time}s of physics ({steps} steps) in {self.solver.dim}D...")
        
        self.solver.initialize_grid()
        
        for _ in range(steps):
            self.solver.step()
            
            # Store copy (Works for both 1D arrays and 2D matrices)
            self.history.append(self.solver.u_curr.copy())
            self.time_steps.append(t)
            t += self.solver.dt
            
        print("Simulation complete.")

    def create_animation(self, skip_frames=10, filename=None):
        """Generates the interactive Plotly figure for 1D (Line) or 2D (Surface)."""
        
        if not self.history:
            print("No data! Run .run() first.")
            return

        # Subsample data
        display_data = self.history[::skip_frames]
        
        # --- 2. Define Plotly Structure based on Dimension ---
        frames = []
        initial_data = []
        layout_settings = {}

        print('Animating.')

        # === 1D SETUP (Scatter) ===
        if self.solver.dim == 1:
            # Base Trace
            initial_data = [go.Scatter(
                x=self.x_axis, 
                y=display_data[0], 
                mode="lines", 
                line=dict(color='royalblue', width=2)
            )]
            
            # Layout
            layout_settings = go.Layout(
                title=f"1D Wave Simulation (T={self.total_time}s)",
                xaxis=dict(title="Position x (m)", range=[0, self.solver.L]),
                yaxis=dict(title="Amplitude", range=[-1.5, 1.5]),
                template="plotly_white"
            )

            # Frame Generation
            for i, wave_state in enumerate(display_data):
                frames.append(go.Frame(
                    data=[go.Scatter(y=wave_state)], # Update y only
                    name=f"f{i}"
                ))

        # === 2D SETUP (Surface) ===
        elif self.solver.dim == 2:
            # Base Trace
            initial_data = [go.Surface(
                x=self.x_axis,
                y=self.y_axis,
                z=display_data[0],
                colorscale='viridis',
                cmin=-1.0, cmax=1.0  # Fix color range so it doesn't flicker
            )]
            
            # Layout (Scene is required for 3D)
            layout_settings = go.Layout(
                title=f"2D Wave Simulation (T={self.total_time}s)",
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Amplitude', range=[-1.5, 1.5]),
                    aspectratio=dict(x=1, y=1, z=0.7) # Adjust visual proportions
                ),
                template="plotly_white"
            )

            # Frame Generation
            for i, wave_state in enumerate(display_data):
                frames.append(go.Frame(
                    data=[go.Surface(z=wave_state)], # Update z only
                    name=f"f{i}"
                ))

        # --- 3. Assemble and Return Figure ---
        
        # Common Animation Controls (Play/Pause)
        updatemenus = [dict(
            type="buttons",
            showactive=False,
            x=0.1, y=0, xanchor="right", yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=20, redraw=True), fromcurrent=True)]),
                dict(label="|| Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
            ]
        )]

        layout_settings.updatemenus = updatemenus
        
        fig = go.Figure(data=initial_data, layout=layout_settings)
        fig.frames = frames

        if filename:
            fig.write_html(filename)
            print(f"Animation saved to {filename}")
        
        return fig