import os
from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from io import BytesIO
import base64

app = Flask(_name_)

@app.post("/generate")
def generate():
    # --- 1. GET DATA FROM FLUTTER APP ---
    try:
        data = request.get_json()
        if data is None:
            data = {} # Handle empty request
    except Exception as e:
        print(f"Could not parse JSON: {e}")
        data = {}

    # Extract scanResults. Default to empty list if not found.
    scanResults = data.get('scanResults', []) # <-- NEW
    customerName = data.get('customerName', '') # <-- NEW
    surveyorName = data.get('surveyorName', '') # <-- NEW

    print(f"Received {len(scanResults)} scan results.") # <-- NEW

    # === 2. CONFIGURATION (Same as before) ===
    np.random.seed(678) 
    depth_min, depth_max = 0, 80
    surface_x_min, surface_x_max = 1, 6
    n_points = 800  
    grid_resolution = 80 
    cbar_ticks = [55, 71, 93, 121, 157, 205, 267, 348, 453, 590, 768, 1000]

    # === 3. BASE DATA GENERATION (Same as before) ===
    # This creates the "rock" and "sand" layers
    depth_grid = np.linspace(depth_min, depth_max, grid_resolution)
    surface_x_grid = np.linspace(surface_x_min, surface_x_max, grid_resolution)
    X_grid, Z_grid = np.meshgrid(surface_x_grid, depth_grid)
    points_x = np.random.uniform(surface_x_min, surface_x_max, n_points)
    points_z = np.random.uniform(depth_min, depth_max, n_points)
    points = np.vstack((points_x, points_z)).T
    values = np.zeros(n_points)

    boundary_z_upper = (40 + 15 * np.sin(points_x * np.pi / 2.5) + 10 * np.random.randn(n_points) * 0.5)
    boundary_z_upper = np.clip(boundary_z_upper, 25, 50)

    is_hard_rock_zone = points_z < boundary_z_upper
    values[is_hard_rock_zone] = 800 + 350 * np.random.rand(np.sum(is_hard_rock_zone))

    is_deep_zone = points_z >= boundary_z_upper
    values[is_deep_zone] = 450 + 250 * np.random.rand(np.sum(is_deep_zone)) 

    # === 4. DYNAMIC POCKET GENERATION (Replaces old hardcoded pockets) ===
    
    # This mask will track all points that become "pockets"
    is_pocket_zone = np.zeros(n_points, dtype=bool) # <-- NEW

    if scanResults and len(scanResults) > 0:
        print("Using scanResults to generate water pockets.") # <-- NEW
        
        # We will space the pockets out along the x-axis
        num_pockets = len(scanResults)
        # Calculate spacing for each pocket, giving them room
        x_spacing = (surface_x_max - surface_x_min - 1) / num_pockets
        current_x_start = surface_x_min + 0.5

        for result in scanResults: # <-- NEW
            try:
                # Get data from the app's JSON
                depth = float(result.get('depth', 0)) # <-- NEW
                res = float(result.get('resistivity', 100)) # <-- NEW
                
                # Define the pocket boundaries based on the data
                pocket_z_min = max(depth_min, depth - 5) # <-- NEW
                pocket_z_max = min(depth_max, depth + 5) # <-- NEW
                pocket_x_min = current_x_start # <-- NEW
                pocket_x_max = current_x_start + (x_spacing * 0.8) # 80% of space # <-- NEW

                # Find all 800 random points that fall inside this new pocket
                mask = (points_z >= pocket_z_min) & (points_z <= pocket_z_max) & \
                       (points_x >= pocket_x_min) & (points_x <= pocket_x_max) # <-- NEW
                
                # Add the original randomness factor
                mask &= (np.random.rand(n_points) > 0.4) # <-- NEW
                
                # Set the value for these points based on the app's resistivity
                # We add a little noise so it's not a flat color
                values[mask] = res + (10 * np.random.rand(np.sum(mask))) # <-- NEW
                
                # Add this pocket's points to the total pocket mask
                is_pocket_zone = is_pocket_zone | mask # <-- NEW
                
                # Move to the next x-position for the next pocket
                current_x_start += x_spacing # <-- NEW

            except ValueError as e:
                print(f"Could not parse result: {result} - Error: {e}") # <-- NEW
            except Exception as e:
                print(f"Error processing result: {e}") # <-- NEW

    else:
        # --- FALLBACK: If no scanResults, use the original random pockets ---
        print("No scanResults found, using original random pockets.") # <-- NEW
        is_pocket_1 = is_deep_zone & (points_x > 1.5) & (points_x < 3.5) & (points_z > 50) & (points_z < 70)
        is_pocket_1 &= (np.random.rand(n_points) > 0.4) 
        values[is_pocket_1] = 50 + 70 * np.random.rand(np.sum(is_pocket_1))
        is_pocket_zone = is_pocket_zone | is_pocket_1 # <-- NEW

        is_pocket_2 = is_deep_zone & (points_x > 4.5) & (points_x < 6.0) & (points_z > 55) & (points_z < 75)
        is_pocket_2 &= (np.random.rand(n_points) > 0.4) 
        values[is_pocket_2] = 100 + 100 * np.random.rand(np.sum(is_pocket_2))
        is_pocket_zone = is_pocket_zone | is_pocket_2 # <-- NEW


    # --- 5. FINAL DATA PREP (Slightly modified) ---
    is_deepest_layer = points_z >= 75
    
    # Make sure the deep layer does NOT overwrite our new pockets
    values[is_deepest_layer & ~is_pocket_zone] = 600 + 300 * np.random.rand(np.sum(is_deepest_layer & ~is_pocket_zone)) # <-- MODIFIED

    values += 100 * np.random.randn(n_points) 

    data_interpolated = griddata(points, values, (X_grid, Z_grid), method='linear')
    data_interpolated = np.clip(data_interpolated, 55, 1000)

    # === 6. PLOTTING (Modified to use dynamic title) ===
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.1)
    ax_map = fig.add_subplot(gs[0, 0])
    levels = np.linspace(55, 1000, 15) 
    c = ax_map.contourf(X_grid, Z_grid, data_interpolated, levels=levels, cmap='jet')

    # axis setup
    ax_map.set_xlabel('Surface-X (m/f)', fontsize=12)
    ax_map.set_ylabel('Depth-Z (m/f)', fontsize=12)
    ax_map.invert_yaxis() 
    ax_map.set_yticks(np.arange(0, 90, 10)) 
    ax_map.set_xticks(np.arange(1, 7, 1))
    
    # --- DYNAMIC TITLE ---
    plot_title = "Earth Depth Profile" # <-- NEW
    if customerName: # <-- NEW
        plot_title += f"\nCustomer: {customerName}" # <-- NEW
    if surveyorName: # <-- NEW
        plot_title += f"   Surveyor: {surveyorName}" # <-- NEW
        
    ax_map.set_title(plot_title, fontsize=16, weight='bold', pad=15) # <-- MODIFIED

    # labels above diagram
    ax_map.text(2.0, -5, 'Soft Rock\nAnd Dry Sand', fontsize=10, color='black', ha='center', va='center')
    ax_map.text(5.0, -5, 'Wet Nature', fontsize=10, color='black', ha='center', va='center')
    ax_map.text(0.7, 20, 'More Hard', fontsize=10, color='black', rotation=90, va='center', ha='center')
    ax_map.text(2.0, 85, 'Most Hard\nStructure', fontsize=10, color='black', ha='center', va='center')
    ax_map.text(3.5, 85, 'Wet Condition', fontsize=10, color='black', ha='center', va='center')
    ax_map.text(5.0, 85, 'Water Bearing Rock', fontsize=10, color='black', ha='center', va='center')

    # legend panel
    ax_legend = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(c, ax_legend, ticks=cbar_ticks, fraction=1.0, pad=0.0)
    cbar.set_label('Value', visible=False) # Hide generic label
    ax_legend.set_yticklabels([]) # Hide default ticks
    ax_legend.set_xticks([]) # Hide x-axis ticks/labels
    ax_legend.set_title('Legend', fontsize=12, pad=10) # <-- NEW: Cleaned up legend

    legend_text_map = {
        'Hard Rock': 900,
        'Medium Hard Rock': 750,
        'Less Medium Rock, Below Soft Rock': 650,
        'Rock, Soil and Wet Nature': 350,
        'Less Dense Porous Rock': 190,
        'Little More Dense Porous Rock': 130,
        'More Dense Porous Rock (Water Bearing Rock Layer)': 85
    }

    for text, y_pos in legend_text_map.items():
        ax_legend.text(1.2, y_pos, text, transform=ax_legend.transData,
                       fontsize=9, ha='left', va='center')
    
        ax_legend.hlines(y_pos, 0.95, 1.15, colors='black', lw=1, transform=ax_legend.get_yaxis_transform())
    
        ax_legend.hlines(y_pos, 0.1, 0.2, colors='black', lw=2, transform=ax_legend.get_yaxis_transform())


    fig.text(0.5, 0.05, 
             'The Earth Depth Profile describes the spread of Soft rock, Hard rock and the Water Bearing Porous rock information.',
             fontsize=12, color='black', ha='center', va='top', wrap=True, transform=fig.transFigure)


    # === 7. CONVERT AND SEND (Same as before) ===
    # convert to PNG memory buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # convert to base64
    image_base64 = base64.b64encode(buf.read()).decode()

    return jsonify({"image": image_base64})

@app.get("/")
def home():
    return "Thermal AI Generator Running!"

if _name_ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
