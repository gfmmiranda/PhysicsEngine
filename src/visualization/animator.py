import numpy as np
import plotly.graph_objects as go

class PhysicsAnimator:
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
        if self.solver.domain.ndim == 1:
            self.x_axis = np.linspace(0, self.solver.domain.L[0], self.solver.domain.N[0])
            self.y_axis = None
        
        elif self.solver.domain.ndim == 2:
            self.x_axis = np.linspace(0, self.solver.domain.L[0], self.solver.domain.N[0])
            self.y_axis = np.linspace(0, self.solver.domain.L[1], self.solver.domain.N[1])

    def run(self):
        """Runs the physics loop and stores data."""
        t = 0.0
        steps = int(self.total_time / self.solver.dt)
        
        print(f"Simulating {self.total_time}s of physics ({steps} steps) in {self.solver.domain.ndim}D...")
        
        for _ in range(steps):
            self.solver.step()
            
            # Store copy (Works for both 1D arrays and 2D matrices)
            u_curr = self.solver.u_curr.copy()
            u_curr[~self.solver.domain.mask] = np.nan
            self.history.append(u_curr)
            self.time_steps.append(t)
            t += self.solver.dt
            
        print("Simulation complete.")

    def create_animation(self, skip_frames=10, filename=None):
        """Generates the interactive Plotly figure with dynamic Z-scaling."""
        
        if not self.history:
            print("No data! Run .run() first.")
            return

        # Subsample data
        display_data = self.history[::skip_frames]
        
        # --- NEW: Calculate Dynamic Limits ---
        # We stack the frames to find the global min/max across the entire animation
        # Use nanmin/nanmax to ignore the NaNs you set for the boundaries/mask
        stack = np.array(display_data)
        global_min = np.nanmin(stack)
        global_max = np.nanmax(stack)
        
        # Safety fallback: If simulation is pure silence (all zeros), set dummy limits
        if global_max == global_min:
            global_max += 1.0
            global_min -= 1.0
            print("Warning: Simulation appears to be flat (min == max).")
        else:
            print(f"Dynamic Scale Found: [{global_min:.2e}, {global_max:.2e}]")

        # Add 10% padding so the wave peaks don't touch the top/bottom of the box
        padding = (global_max - global_min) * 0.1
        z_limits = [global_min - padding, global_max + padding]
        # -------------------------------------

        # Check for listeners safely
        listeners = getattr(self.solver, 'listeners', [])
        has_listeners = len(listeners) > 0

        frames = []
        initial_data = []
        layout_settings = {}

        print('Animating...')

        # === 1D SETUP ===
        if self.solver.domain.ndim == 1:
            initial_data.append(go.Scatter(
                x=self.x_axis, y=display_data[0], mode="lines", name="Wave",
                line=dict(color='royalblue', width=2)
            ))
            
            if has_listeners:
                l_x = [l.pos[0] for l in listeners]
                l_y = [display_data[0][l.grid_idx] for l in listeners]
                initial_data.append(go.Scatter(
                    x=l_x, y=l_y, mode="markers", name="Listener",
                    marker=dict(color='red', size=12, symbol='x')
                ))
            
            layout_settings = go.Layout(
                title=f"1D Simulation (Range: {global_min:.2e} to {global_max:.2e})",
                xaxis=dict(title="Position (m)", range=[0, self.solver.domain.L[0]]),
                yaxis=dict(title="Amplitude", range=z_limits), # <--- Dynamic Range applied
                template="plotly_white"
            )

            for i, state in enumerate(display_data):
                frame_data = [go.Scatter(y=state)]
                if has_listeners:
                    l_y_new = [state[l.grid_idx] for l in listeners]
                    frame_data.append(go.Scatter(y=l_y_new))
                frames.append(go.Frame(data=frame_data, name=f"f{i}"))

        # === 2D SETUP ===
        elif self.solver.domain.ndim == 2:
            initial_data.append(go.Surface(
                x=self.x_axis, y=self.y_axis, z=display_data[0],
                colorscale='viridis',
                cmin=global_min, cmax=global_max, # <--- Dynamic Color Scale
                name="Wave"
            ))
            
            if has_listeners:
                l_x = [l.pos[0] for l in listeners]
                l_y = [l.pos[1] for l in listeners]
                l_z = [display_data[0][l.grid_idx] for l in listeners]
                
                initial_data.append(go.Scatter3d(
                    x=l_x, y=l_y, z=l_z, mode="markers", name="Listener",
                    marker=dict(color='red', size=5, symbol='circle')
                ))

            layout_settings = go.Layout(
                title=f"2D Simulation (Range: {global_min:.2e} to {global_max:.2e})",
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Amplitude', range=z_limits), # <--- Dynamic Z-Axis
                    aspectratio=dict(x=1, y=1, z=0.7)
                ),
                template="plotly_white"
            )

            for i, state in enumerate(display_data):
                frame_data = [go.Surface(z=state)]
                if has_listeners:
                    l_z_new = [state[l.grid_idx] for l in listeners]
                    l_x = [l.pos[0] for l in listeners]
                    l_y = [l.pos[1] for l in listeners]
                    frame_data.append(go.Scatter3d(x=l_x, y=l_y, z=l_z_new))
                frames.append(go.Frame(data=frame_data, name=f"f{i}"))

        # --- Final Assembly ---
        updatemenus = [dict(
            type="buttons", showactive=False,
            x=0.1, y=0, xanchor="right", yanchor="top", pad=dict(t=0, r=10),
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