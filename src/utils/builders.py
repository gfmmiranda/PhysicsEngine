import numpy as np

def build_smart_speaker(domain, center, width=1.0, height=1.0, wall_thickness=0.1, material_value=1.0):
    """
    Constructs a U-shaped speaker enclosure in the given domain.
    """
    cx, cy = center
    
    # Dimensions
    w_out = width + 2*wall_thickness
    h_out = height + wall_thickness
    
    # 1. Back Wall
    domain.add_rectangular_obstacle(
        pos=[cx, cy - height/2 - wall_thickness/2], 
        size=[w_out, wall_thickness]
    )
    
    # 2. Left Wall
    domain.add_rectangular_obstacle(
        pos=[cx - width/2 - wall_thickness/2, cy], 
        size=[wall_thickness, height]
    )
    
    # 3. Right Wall
    domain.add_rectangular_obstacle(
        pos=[cx + width/2 + wall_thickness/2, cy], 
        size=[wall_thickness, height]
    )

def build_line_array(domain, center, num_elements, spacing, frequency, axis='y'):
    """Adds a line of HarmonicSources to the domain."""
    from src.components import HarmonicSource
    
    cx, cy = center
    length = (num_elements - 1) * spacing
    start = -length / 2.0
    
    sources = []
    for i in range(num_elements):
        offset = start + i * spacing
        pos = [cx + offset, cy] if axis == 'x' else [cx, cy + offset]
        
        s = HarmonicSource(pos=pos, frequency=frequency)
        domain.add_source(s)
        sources.append(s)
    
    return sources

def build_double_slit(domain, wall_x, slit_width, slit_separation, wall_thickness=0.2):
    """
    Constructs a barrier with two slits.
    
    Parameters
    ----------
    wall_x : float
        X-position of the wall.
    slit_width : float
        Size of the gap (aperture).
    slit_separation : float
        Center-to-center distance between slits.
    """
    # Domain dimensions
    Ly = domain.L[1]
    cy = Ly / 2.0 # Center Y
    
    t = wall_thickness
    
    # Calculate Y-coordinates for the gaps
    # Slit 1 (Top)
    y_slit1_center = cy + slit_separation / 2.0
    y_slit1_bottom = y_slit1_center - slit_width / 2.0
    y_slit1_top    = y_slit1_center + slit_width / 2.0
    
    # Slit 2 (Bottom)
    y_slit2_center = cy - slit_separation / 2.0
    y_slit2_bottom = y_slit2_center - slit_width / 2.0
    y_slit2_top    = y_slit2_center + slit_width / 2.0
    
    # --- Build 3 Wall Segments ---
    
    # 1. Bottom Segment (Floor to Slit 2)
    h_bot = y_slit2_bottom
    domain.add_rectangular_obstacle(
        pos=[wall_x, h_bot/2.0],
        size=[t, h_bot]
    )
    
    # 2. Middle Segment (Between Slits)
    h_mid = y_slit1_bottom - y_slit2_top
    y_mid = y_slit2_top + h_mid/2.0
    domain.add_rectangular_obstacle(
        pos=[wall_x, y_mid],
        size=[t, h_mid]
    )
    
    # 3. Top Segment (Slit 1 to Ceiling)
    h_top = Ly - y_slit1_top
    y_top = y_slit1_top + h_top/2.0
    domain.add_rectangular_obstacle(
        pos=[wall_x, y_top],
        size=[t, h_top]
    )
    
    print(f"✅ Double Slit built at x={wall_x} (Sep: {slit_separation}m, Width: {slit_width}m)")

def build_parabolic_reflector(domain, vertex, focus_dist, aperture, wall_thickness=0.1):
    """
    Constructs a parabolic dish opening to the right.
    
    Parameters
    ----------
    vertex : list
        [x, y] coordinates of the bottom of the dish.
    focus_dist : float
        Distance from vertex to focal point (f).
    aperture : float
        Total height (diameter) of the dish.
    """
    vx, vy = vertex
    f = focus_dist
    t = wall_thickness
    
    # 1. Get Grid Coordinates
    X, Y = domain.grids
    
    # 2. Define Parabola Equation: x = y^2 / 4f + vx
    # We shift Y by vy to center it vertically
    ideal_x = ((Y - vy)**2) / (4.0 * f) + vx
    
    # 3. Create the Mask
    # We select points that are close to the ideal curve (within thickness)
    # AND are within the aperture height
    on_curve = np.abs(X - ideal_x) < (t / 2.0)
    within_aperture = np.abs(Y - vy) <= (aperture / 2.0)
    
    # Apply to Domain
    mask = on_curve & within_aperture
    
    # Set material (Reflective)
    # If using Domain2D with material support:
    if hasattr(domain, 'set_material'):
        domain.materials[mask] = 1.0 # Hard wall
        
    domain.mask[mask] = False
    domain.update_boundaries()
    
    # Calculate and print the focal point for the user
    focal_point = [vx + f, vy]
    print(f"✅ Parabolic Reflector built.")
    print(f"   Vertex: {vertex}")
    print(f"   Focal Point: {focal_point} <--- PLACE SOURCE HERE")
    
    return focal_point