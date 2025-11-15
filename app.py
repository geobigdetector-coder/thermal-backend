import os
from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg') # Set backend for server
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from io import BytesIO
import base64

# --- Imports from your new script ---
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter, minimum_filter
from matplotlib import patheffects
from matplotlib.patches import Patch
# --- End new imports ---

app = Flask(__name__)

@app.get("/")
def home():
    return "Python Plot Generator Running!"

@app.post("/generate")
def generate():
    
    # === 1. GET DYNAMIC DATA FROM FLUTTER APP ===
    try:
        data = request.get_json()
        if data is None: data = {}
    except Exception as e:
        print(f"Could not parse JSON: {e}")
        data = {}

    # Extract dynamic values, using your script's values as defaults
    
    # --- DYNAMIC SITE NAME ---
    customer = data.get('customerName', 'N/A')
    surveyor = data.get('surveyorName', 'N/A')
    site_name = f"Customer: {customer}, Surveyor: {surveyor}"
    
    # --- DYNAMIC LOCATION ---
    location = data.get('location', {})
    latitude = float(location.get('latitude', 18.4645343)) # Default from your script
    longitude = float(location.get('longitude', 73.8294))  # Default from your script

    # --- DYNAMIC DEPTH ---
    depth_min = 0
    depth_max = int(data.get('selectedDepthFt', 200)) # Use app's max depth, default 200
    
    # --- DYNAMIC WATER LAYERS (THE 3 DEPTHS) ---
    scanResults = data.get('scanResults', [])
    water_layers = []
    if scanResults:
        for result in scanResults:
            try:
                # Add the depth from each scan result
                water_layers.append(float(result.get('depth')))
            except (ValueError, TypeError):
                pass # Ignore if depth is not a valid number
    
    # Fallback: If app sent no depths, use your script's defaults
    if not water_layers:
        water_layers = [25, 100, 160] # Default from your script

    print(f"Generating plot for {site_name} with max_depth={depth_max} and layers at {water_layers}")

    # =================================================================
    # === YOUR PLOTTING SCRIPT (UNCHANGED UI) ===
    # =================================================================

    # === CONFIGURATION ===
    np.random.seed(42)
    # depth_min, depth_max = 0, 200  (NOW DYNAMIC)
    surface_x_min, surface_x_max = 1, 7
    grid_res = 400

    # Geological & aquifer configuration
    # water_layers = [25, 100, 160] (NOW DYNAMIC)
    fracture_intensity = 0.28
    dip_angle_deg = 7
    dip_tilt_factor = np.tan(np.radians(dip_angle_deg))

    # === GRID ===
    depth_grid = np.linspace(depth_min, depth_max, grid_res)
    surface_x_grid = np.linspace(surface_x_min, surface_x_max, grid_res)
    X, Z = np.meshgrid(surface_x_grid, depth_grid)

    # === APPLY EASTWARD DIP ===
    Z_tilted = Z + dip_tilt_factor * (X - surface_x_min) * (depth_max / (surface_x_max - surface_x_min)) * 0.3

    # === GEOLOGY PATTERN ===
    def generate_geology(X, Z, freq=0.07, seed=1):
        np.random.seed(seed)
        base = (
            0.6 * np.sin(X * freq * np.pi)
            + 0.3 * np.cos(Z * freq * np.pi)
            + 0.25 * np.sin((X + Z) * freq * 1.5)
            + 0.15 * np.sin((X - Z) * freq * 2.3)
        )
        noise = np.random.normal(0, fracture_intensity, size=base.shape)
        return gaussian_filter(base + noise, sigma=5)

    pattern = generate_geology(X, Z_tilted, freq=0.06, seed=3)
    data = 550 + 450 * pattern
    data = np.clip(data, 55, 1000)

    # === ADD WATER ZONES (DETECTED POINTS) ===
    for i, depth in enumerate(water_layers):
        organic_mask = (
            np.sin((X * 0.7 + np.random.rand() * 2) * np.pi / 3)
            + np.cos((Z * 0.5 + i * 3) * np.pi / 6)
            + np.random.normal(0, 0.3, size=X.shape)
        )
        organic_mask = gaussian_filter(organic_mask, sigma=8)
        dip_correction = dip_tilt_factor * (X - surface_x_min) * 15
        depth_tilted = depth + dip_correction

        horizontal_spread = np.exp(-((Z - depth_tilted) ** 2) / (2 * (6 ** 2)))
        lateral_spread = np.exp(-((X - np.random.uniform(3, 5)) ** 2) / (2 * (2.5 ** 2)))
        stretched_zone = horizontal_spread * lateral_spread * (organic_mask > 0.3)

        water_blend = gaussian_filter(stretched_zone.astype(float), sigma=3)
        data -= 500 * water_blend
        data = np.clip(data, 55, 1000)

    # === 3D SHADING ===
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(data, cmap=plt.cm.jet, vert_exag=0.1, blend_mode='soft')

    # === PLOT ===
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgb, extent=[surface_x_min, surface_x_max, depth_max, depth_min], aspect='auto')

    # === ADD CONTOURS ===
    contour_levels = np.linspace(100, 900, 10)
    cs = ax.contour(X, Z, data, levels=contour_levels, colors='black', linewidths=0.4, alpha=0.35)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%d', colors='black')

    # === COLORBAR ===
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=55, vmax=1000))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.08, orientation='horizontal', shrink=0.7)
    cbar.set_label("Resistivity (Ω·m)", fontsize=11)

    # === GEOLOGICAL LABELS ===
    ax.text(2.0, 20, "Hard Rock", color="darkred", fontsize=11, fontweight="bold")
    ax.text(2.5, 60, "Soft Rock", color="orange", fontsize=11, fontweight="bold")
    ax.text(4.0, 130, "Moist Zone", color="green", fontsize=11, fontweight="bold")
    ax.text(4.5, 190, "Water-Bearing Layer", color="navy", fontsize=11, fontweight="bold")

    # === SCALE BARS ===
    ax.plot([1, 1], [0, 200], color='black', lw=1.2)
    ax.text(0.5, 200, "Depth (ft)", rotation=90, fontsize=9, color="black", va="top")
    ax.hlines(y=210, xmin=1, xmax=7, color='black', lw=1.2)
    ax.text(3.5, 214, "Surface Distance (~6 m)", fontsize=9, ha="center")

    # === LEGEND (Moved to Right Side) ===
    legend_patches = [
        Patch(color='darkred', label='Hard Rock (High Resistivity)'),
        Patch(color='orange', label='Soft Rock / Sand (Medium Resistivity)'),
        Patch(color='green', label='Moist Zone (Moderate Resistivity)'),
        Patch(color='blue', label='Water-Bearing Layer (Low Resistivity)'),
    ]

    legend = ax.legend(
        handles=legend_patches,
        loc='upper right',
        bbox_to_anchor=(1.0, 0.25),
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=False,
        borderpad=0.8,
        labelspacing=0.7
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.set_zorder(20)

    # === DETECTED WATER POINTS (Pixel-Perfect: Darkest Blue Only) ===
    for i, target_depth in enumerate(water_layers, start=1):
        # Define search window: tight vertical range (±4 ft), full x-range
        z_min = max(0, target_depth - 4)
        z_max = min(depth_max, target_depth + 4)
        z_mask = (Z >= z_min) & (Z <= z_max)
        
        # CRITICAL: Only consider pixels below 180 Ω·m (darkest blue threshold)
        dark_blue_mask = (data <= 180) & z_mask
        
        if np.any(dark_blue_mask):
            # Get coordinates of ALL darkest pixels
            dark_pixels = data[dark_blue_mask]
            z_coords = Z[dark_blue_mask]
            x_coords = X[dark_blue_mask]
            
            # Find absolute minimum resistivity in this zone
            min_val = np.min(dark_pixels)
            min_indices = np.where(dark_pixels == min_val)[0]
            
            # If multiple darkest pixels, pick the one closest to center x=4.0
            if len(min_indices) > 1:
                distances = np.abs(x_coords[min_indices] - 4.0)
                best_idx = min_indices[np.argmin(distances)]
            else:
                best_idx = min_indices[0]
            
            x_point = x_coords[best_idx]
            z_point = z_coords[best_idx]
        else:
            # Fallback: find global minimum in vertical window
            window_data = np.where(z_mask, data, np.inf)
            min_idx = np.unravel_index(np.argmin(window_data), window_data.shape)
            x_point = X[min_idx]
            z_point = Z[min_idx]

        # PLOT MARKER AT EXACT DARKEST PIXEL
        ax.scatter(x_point, z_point, s=90, color='cyan', edgecolors='white',
                   linewidths=1.2, zorder=12, marker='*', alpha=0.95)
        
        # ENHANCED LABEL PLACEMENT
        x_label = min(x_point + 1.3, 6.3)
        z_label = z_point
        
        # SUBTLE CONNECTOR (cyan for visibility)
        ax.plot([x_label - 0.15, x_point], [z_label, z_point],
                color='cyan', lw=1.1, ls='-', alpha=0.85, zorder=11)
        
        # BOLD LABEL WITH WHITE OUTLINE (FIXED SYNTAX)
        ax.text(x_label, z_label,
                f"Water Point {i} : {int(target_depth)} ft",
                color="navy", fontsize=10.5, weight="bold",
                va="center", ha="left", zorder=12,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground='white')])

    # === TITLES (NOW DYNAMIC) ===
    ax.set_title(f"Earth Depth Profile – {site_name}\nLat: {latitude:.6f}, Lon: {longitude:.6f}\n7° Eastward Dip & Resistivity Map",
                 fontsize=14, weight="bold", pad=14)
    ax.set_xlabel("Surface-X (m/f)", fontsize=11)
    ax.set_ylabel("Depth-Z (ft)", fontsize=11)
    ax.invert_yaxis()

    # === INTERPRETATION TABLE ===
    table_text = (
        "Resistivity Interpretation Table\n"
        "────────────────────────────────────────\n"
        "Value Range | Interpretation | Typical Rock Type\n"
        "100–200  →  Very low resistivity → Water-bearing, wet clay, fractured rock\n"
        "300–400  →  Low–medium resistivity → Moist sand, partially saturated soil\n"
        "500–700  →  Moderate resistivity → Compact soil, semi-hard rock, dry layers\n"
        "800–900  →  High resistivity → Hard igneous rock, dry basement"
    )

    plt.figtext(
        0.5, 0.03,
        table_text,
        wrap=True,
        horizontalalignment='center',
        fontsize=9,
        fontfamily='monospace',
        bbox=dict(facecolor='white', alpha=0.95, edgecolor='gray', boxstyle='round,pad=0.5')
    )

    # Final layout adjustments
    plt.subplots_adjust(bottom=0.2, top=0.93, right=0.82)
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # =================================================================
    # === SERVER CONVERSION (No UI Change) ===
    # =================================================================
    
    # Convert plot to PNG in memory
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150) # Use fig.savefig
    plt.close(fig) # Must close figure to prevent memory leak
    buf.seek(0)

    # Encode PNG to base64
    image_base64 = base64.b64encode(buf.read()).decode()

    # Return as JSON
    return jsonify({"image": image_base64})

# --- Main server runner ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

