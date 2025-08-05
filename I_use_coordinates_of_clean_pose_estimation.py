
""" use the cleaned .csv files (pose estimation) """
""" get behavior: freezing if possible """
""" Get velocity """
#%% Load all_dfs from pose_estimation_clean/
import pandas as pd
from pathlib import Path

# --- Folder where cleaned CSVs are stored ---
input_folder = Path('clean_pose_estimation')
all_dfs = {}

# --- Load all CSVs into dictionary ---
for file in sorted(input_folder.glob("*.csv")):
    key = file.stem  # e.g., '994_s4_rm'
    print(f"Loading: {file.name}")
    df = pd.read_csv(file)
    all_dfs[key] = df


#%%
"""Plot All 12 Body Part Trajectories (One Session)"""
""" 2D trajectory plot """

# %%
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_trajectories_from_csv(filename, folder='clean_pose_estimation', save_folder='trajectory_plot'):
    """Plot and save x vs y trajectories for all body parts in custom order."""
    filepath = os.path.join(folder, filename)
    df = pd.read_csv(filepath)
    os.makedirs(save_folder, exist_ok=True)

    # Drop 'bodyparts_coords' column if present
    if 'bodyparts_coords' in df.columns:
        df = df.drop(columns=['bodyparts_coords'])

    # Extract all bodypart base names
    all_parts = sorted(set(col.rsplit('_', 1)[0] for col in df.columns if '_x' in col))

    # Define custom order
    first = ['1_snout']
    last = ['10_tail_base', '11_tail_mid', '12_tail_end']
    middle = [part for part in all_parts if part not in first + last]
    ordered_parts = first + sorted(middle) + last

    # Plot layout
    n = len(ordered_parts)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    # Plot each trajectory
    for i, part in enumerate(ordered_parts):
        x = df[f"{part}_x"]
        y = df[f"{part}_y"]
        axes[i].plot(x, y, lw=1)
        axes[i].set_title(part.upper())
        axes[i].invert_yaxis()
        axes[i].set_aspect('equal')

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis('off')

    plt.suptitle(f"Trajectories for {filename}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save as PNG and SVG with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_root = os.path.splitext(filename)[0]
    png_path = os.path.join(save_folder, f"{name_root}_{timestamp}.png")
    svg_path = os.path.join(save_folder, f"{name_root}_{timestamp}.svg")
    #uncomment for saving:
    # plt.savefig(png_path, dpi=300)
    # plt.savefig(svg_path)
    plt.close()

    print(f"Saved: {os.path.basename(png_path)} and .svg")

# %%
input_folder = "clean_pose_estimation"
for file in os.listdir(input_folder):
    if file.endswith('.csv'):
        print(f"Plotting {file}...")
        plot_trajectories_from_csv(file, folder=input_folder)

# trajectories plotted and saved 
# %%


""" """
