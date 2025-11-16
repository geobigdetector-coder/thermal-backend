import io
import base64
from flask import Flask, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Patch
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-GUI server environment

# === FLASK APP INITIALIZATION ===
app = Flask(__name__)

# === MAIN PLOTTING FUNCTION (ADAPTED FROM YOUR SCRIPT) ===
def create_geology_plot(report_data):
    """
    Generates the earth depth profile plot based on dynamic report data.
    """
    
    # === 1. EXTRACT DYNAMIC DATA FROM THE FLUTTER APP'S JSON ===
    
    # Site metadata
    site_name = report_data.get('customerName', 'Survey Site')
    location = report_data.get('location', {})
    try:
        latitude = float(location.get('latitude', 18.4645343))
    except (ValueError, TypeError):
        latitude = 18.4645343
    try:
        longitude = float(location.get('longitude', 73.8294))
    except (ValueError, TypeError):
        longitude = 73.8294

    # Configuration
    try:
        depth_max = int(report_data.get('selectedDepthFt', 200))
    except (ValueError, TypeError):
        depth_max = 200
        
    depth_min = 0
    surface_x_min, surface_x_max = 1, 7
    
    # --- OPTIMIZATION 1: Reduced grid resolution ---
    grid_res = 200 # Was 400. This saves 75% of memory for numpy arrays.
    # --- END OPTIMIZATION ---

    # Geological & aquifer configuration (from scanResults)
    scan_results = report_data.get('scanResults', [])
    water_layers = []
    resistivity_at_layers = []
    frequency_at_layers = []
    
    for r in scan_results:
        try:
            water_layers.append(int(r['depth']))
            resistivity_at_layers.append(float(r['resistivity']))
            frequency_at_layers.append(float(r['frequency']))
        except (ValueError, TypeError, KeyError):
            continue # Skip invalid records

    # Fallback if no valid scan results are provided
    if not water_layers:
        water_layers = [25, 100, 160] # Default depths
        # Use default res/freq if using default depths
        resistivity_at_layers = [np.random.uniform(55, 150) for _ in water_layers]
        frequency_at_layers = [np.random.uniform(700, 900) for _ in water_layers]

    # Use data from your script's config
    fracture_intensity = 0.28
    dip_angle_deg = 7
    dip_tilt_factor = np.tan(np.radians(dip_angle_deg))
    np.random.seed(42) # Keep seed for reproducible patterns

    # === 2. GEOLOGY & PLOTTING LOGIC (FROM YOUR SCRIPT) ===

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

    # === CREATE FIGURE WITH TWO SUBPLOTS ===
    # Use 'fig' to manage the figure object
    fig = plt.figure(figsize=(12, 13.5), dpi=100) 

    # === PLOT 1: RESISTIVITY MAP (TOP) ===
    ax1 = plt.subplot(2, 1, 1)
    im = ax1.imshow(rgb, extent=[surface_x_min, surface_x_max, depth_min, depth_max], aspect='auto')

    # === ADD CONTOURS ===
    contour_levels = np.linspace(100, 900, 10)
    cs = ax1.contour(X, Z, data, levels=contour_levels, colors='black', linewidths=0.4, alpha=0.35)
    ax1.clabel(cs, inline=True, fontsize=7, fmt='%d', colors='black')

    # === COLORBAR ===
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=55, vmax=1000))
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.035, pad=0.08, orientation='horizontal', shrink=0.7)
    cbar.set_label("Resistivity (Ω·m)", fontsize=11)

    # === GEOLOGICAL LABELS ===
    def physical_to_plot(physical_depth):
        return depth_max - physical_depth # Convert physical depth to plot Y-coord
        
    ax1.text(1.8, physical_to_plot(25), "Hard Rock", color="darkred", fontsize=11, fontweight="bold", 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
    ax1.text(2.2, physical_to_plot(65), "Soft Rock", color="orange", fontsize=11, fontweight="bold",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
    ax1.text(3.7, physical_to_plot(125), "Moist Zone", color="green", fontsize=11, fontweight="bold",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
    # Adjust last label based on max_depth
    ax1.text(4.2, physical_to_plot(min(185, depth_max - 15)), "Water-Bearing Layer", color="navy", fontsize=11, fontweight="bold",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

    # === SCALE BARS ===
    ax1.plot([1, 1], [0, depth_max], color='black', lw=1.2)
    ax1.text(0.7, depth_max/2, "Depth (ft)", rotation=90, fontsize=9, color="black", va="center", ha="center")
    ax1.hlines(y=depth_max + (depth_max * 0.06), xmin=1, xmax=7, color='black', lw=1.2)
    ax1.text(4.0, depth_max + (depth_max * 0.08), "Surface Distance (~6 m)", fontsize=9, ha="center")

    # === LEGEND ===
    legend_patches = [
        Patch(color='darkred', label='Hard Rock (High Resistivity)'),
        Patch(color='orange', label='Soft Rock / Sand (Medium Resistivity)'),
        Patch(color='green', label='Moist Zone (Moderate Resistivity)'),
        Patch(color='blue', label='Water-Bearing Layer (Low Resistivity)'),
    ]
    legend = ax1.legend(
        handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.35),
        fontsize=9, frameon=True, fancybox=True, shadow=False,
        borderpad=0.8, labelspacing=0.7
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.set_zorder(20)

    # === PRECISE WATER POINTS DETECTION ===
    # Use the actual depths and resistivity values from the app
    for i, (target_depth, target_res) in enumerate(zip(water_layers, resistivity_at_layers), start=1):
        target_idx = np.argmin(np.abs(depth_grid - target_depth))
        window_size = 15
        start_idx = max(0, target_idx - window_size)
        end_idx = min(grid_res - 1, target_idx + window_size)
        window_data = data[start_idx:end_idx + 1, :]
        
        # Find the actual minimum in the window, but use the app's resistivity for the label
        min_idx_flat = np.argmin(window_data)
        z_rel_idx, x_idx = np.unravel_index(min_idx_flat, window_data.shape)
        z_idx = start_idx + z_rel_idx
        
        x_point = surface_x_grid[x_idx]
        z_point = depth_grid[z_idx] # This is the "detected" physical depth
        plot_y = depth_max - z_point # This is the Y-coord for plotting
        
        x_point = np.clip(x_point, surface_x_min + 0.3, surface_x_max - 0.3)
        plot_y = np.clip(plot_y, 10, depth_max - 10)
        
        ax1.scatter(x_point, plot_y, s=110, color='white', edgecolors='black',
                    linewidths=1.8, zorder=15, marker='*', alpha=0.95)
        
        label_offset = (i - 2) * 18
        
        if x_point < 4.0:
            x_label = x_point + 1.2
            ha = 'left'
            ax1.plot([x_point, x_label - 0.3], [plot_y, plot_y + label_offset],
                     color='black', lw=1.3, ls='-', alpha=0.8, zorder=14)
        else:
            x_label = x_point - 1.2
            ha = 'right'
            ax1.plot([x_label + 0.3, x_point], [plot_y + label_offset, plot_y],
                     color='black', lw=1.3, ls='-', alpha=0.8, zorder=14)
        
        # Use the app's resistivity data in the label
        ax1.text(x_label, plot_y + label_offset,
                 f"WP{i}: {int(target_depth)} ft\n({int(target_res)} Ω·m)",
                 color="black", fontsize=9.5, weight="bold",
                 va="center", ha=ha, zorder=15,
                 bbox=dict(facecolor='yellow', alpha=0.85, edgecolor='black', boxstyle='round,pad=0.3'))

    # === TITLES & AXES FOR RESISTIVITY PLOT ===
    ax1.set_title(f"Earth Depth Profile – {site_name}\nLat: {latitude:.6f}, Lon: {longitude:.6f}\n7° Eastward Dip & Resistivity Map",
                  fontsize=14, weight="bold", pad=18)
    ax1.set_xlabel("Surface-X (m/f)", fontsize=11)
    ax1.set_ylabel("Depth (ft)", fontsize=11)

    # Set Y-axis ticks to match the max_depth
    tick_interval = 50 if depth_max >= 150 else 25
    y_ticks = np.arange(0, depth_max + 1, tick_interval)
    y_tick_labels = [str(int(depth_max - y)) for y in y_ticks]
    
    ax1.set_ylim(0, depth_max)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)

    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.grid(False)

    # === PLOT 2: FREQUENCY PLOT (BOTTOM) ===
    ax2 = plt.subplot(2, 1, 2)

    # === GENERATE SYNTHETIC FREQUENCY DATA (as in your script) ===
    n_points = 350
    depth_line = np.linspace(depth_min, depth_max, n_points)
    freq = (
        500 + 
        100 * np.sin(depth_line * 0.15 * (200 / depth_max)) + 
        150 * np.sin(depth_line * 0.04 * (200 / depth_max)) + 
        80 * np.sin(depth_line * 0.25 * (200 / depth_max)) +
        60 * np.sin(depth_line * 0.4 * (200 / depth_max)) +
        30 * np.random.randn(n_points)
    )
    for i in range(1, len(freq) - 1):
        if np.random.rand() > 0.95:
            freq[i] = 0.5 * (freq[i-1] + freq[i+1]) + np.random.uniform(-150, 150)
    freq = np.clip(freq, 0, 1000)

    # Add specific peaks at water layer depths, using app's freq data
    for (d, f) in zip(water_layers, frequency_at_layers):
        if d <= depth_max: # Only plot if within range
            idx = np.argmin(np.abs(depth_line - d))
            freq[idx] = max(freq[idx], f, 750) # Make it a noticeable peak using app data

    # === PLOT THE FREQUENCY DATA ===
    ax2.plot(depth_line, freq, color='blue', linewidth=1.8)

    # ADD PEAK MARKERS AT WATER LAYERS
    for (d, f) in zip(water_layers, frequency_at_layers):
         if d <= depth_max: # Only plot if within range
            plot_f = freq[np.argmin(np.abs(depth_line - d))] # Get the actual plotted freq
            ax2.scatter(d, plot_f, s=80, color='red', zorder=10, edgecolor='black')
            ax2.text(d + (depth_max * 0.025), plot_f + 20, f"{d}ft", fontsize=9, color='darkred', fontweight='bold')

    # === STYLE SETTINGS ===
    ax2.set_title("Depth Vs Frequency Mapping", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Depth (ft)", fontsize=11)
    ax2.set_ylabel("Freq (MHz)", fontsize=11)
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(depth_min, depth_max)
    ax2.set_ylim(0, 1000)
    ax2.set_xticks(y_ticks) # Use same ticks as Y-axis from plot 1
    ax2.set_yticks([0, 200, 400, 600, 800, 1000])

    # === FINAL LAYOUT ADJUSTMENTS ===
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.subplots_adjust(hspace=0.28, top=0.93)

    # === ADD TABLE ===
    table_data = [
        ["Value Range (Ω·m)", "Interpretation", "Typical Material"],
        ["100-200", "Very low resistivity", "Water-bearing zones, wet clay"],
        ["300-400", "Low-medium resistivity", "Moist sand, fractured rock"],
        ["500-700", "Moderate resistivity", "Dry soil, semi-hard rock"],
        ["800-1000", "High resistivity", "Hard igneous rock, dry basement"]
    ]
    table = plt.table(
        cellText=table_data, cellLoc='center', loc='bottom',
        bbox=[0.05, -0.30, 0.9, 0.15], colWidths=[0.2, 0.4, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#d1e5f0')
            cell.set_text_props(weight='bold', color='navy')
            cell.set_edgecolor('black')
        else:
            cell.set_edgecolor('#888888')
        cell.set_height(0.12)
    plt.figtext(0.5, 0.048, 'Resistivity Interpretation Guide', 
                ha='center', fontsize=11, weight='bold', color='navy')


    # === 3. SAVE PLOT TO IN-MEMORY BUFFER ===
    buf = io.BytesIO()
    
    # --- OPTIMIZATION 2: Reduced DPI for rendering ---
    fig.savefig(buf, format='png', dpi=96, bbox_inches='tight') # Was 300. This is the biggest memory saver.
    # --- END OPTIMIZATION ---

    buf.seek(0)
    
    # === 4. ENCODE TO BASE64 AND CLEAN UP ===
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # IMPORTANT: Close the figure to free up memory
    
    return img_base64


# === FLASK API ENDPOINT ===
@app.route("/generate", methods=["POST"])
def handle_generate():
    """
    This is the API endpoint that the Flutter app will call.
    It receives the JSON report data, generates the plot,
    and returns the Base64-encoded image.
    """
    try:
        # Get the JSON data from the request body
        report_data = request.json
        if not report_data:
            return jsonify({"error": "No JSON data received"}), 400

        # Generate the plot
        base64_image_string = create_geology_plot(report_data)

        # Return the image in the format the app expects
        return jsonify({
            "image": base64_image_string
        })

    except Exception as e:
        print(f"Error generating plot: {e}") # Log the error to the console
        return jsonify({"error": str(e)}), 500

# === RUN THE SERVER ===
if __name__ == '__main__':
    # Runs the server on localhost:5000
    # For production (like on Render), a Gunicorn server is used instead
    app.run(debug=True, host='0.0.0.0', port=5000)
