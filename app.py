# Set the backend for Matplotlib to 'Agg'
# This is CRITICAL for running in a headless server environment
# It must be done BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
from flask import Flask, request, jsonify

# New imports used by your script
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter
from matplotlib import patheffects
from matplotlib.patches import Patch

# Initialize Flask app
app = Flask(__name__)


def create_earth_depth_plot():
    """
    This function contains the advanced plotting logic.
    """

    # === SITE METADATA ===
    site_name = "Tronix365, Pune"
    latitude = 18.4645343
    longitude = 73.8294

    # === CONFIGURATION ===
    np.random.seed(42)
    depth_min, depth_max = 0, 200  # feet
    surface_x_min, surface_x_max = 1, 7  # meters
    grid_res = 400

    # Geological & aquifer configuration
    water_layers = [25, 100, 160]  # depths in feet
    fracture_intensity = 0.28
    dip_angle_deg = 7  # eastward dip
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

    # === ADD WATER ZONES ===
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

    # === CONTOURS ===
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

    # === LEGEND ===
    legend_patches = [
        Patch(color='darkred', label='Hard Rock (High Resistivity)'),
        Patch(color='orange', label='Soft Rock / Sand (Medium Resistivity)'),
        Patch(color='green', label='Moist Zone (Moderate Resistivity)'),
        Patch(color='blue', label='Water-Bearing Layer (Low Resistivity)')
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

    # === TITLES ===
    ax.set_title(
        f"Earth Depth Profile – {site_name}\nLat: {latitude}, Lon: {longitude}\n7° Eastward Dip & Resistivity Map",
        fontsize=14,
        weight="bold",
        pad=14
    )
    ax.set_xlabel("Surface-X (m/f)", fontsize=11)
    ax.set_ylabel("Depth-Z (ft)", fontsize=11)
    ax.invert_yaxis()

    # Layout
    plt.subplots_adjust(bottom=0.2, top=0.93, right=0.82)

    return fig


@app.route("/generate", methods=["POST"])
def generate():
    """
    API endpoint to generate the plot and return it as Base64 PNG.
    """
    try:
        fig = create_earth_depth_plot()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return jsonify({"image": image_base64})

    except Exception as e:
        plt.close('all')
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return "Python PNG Generator is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
