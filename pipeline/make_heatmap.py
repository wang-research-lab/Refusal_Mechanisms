import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def build_similarity_matrix(pairwise):
    components = set()
    for key in pairwise.keys():
        parts = key.split("_vs_")
        if len(parts) == 2:
            components.add(parts[0])
            components.add(parts[1])

    def component_sort_key(name):
        match = re.match(r"([a-zA-Z]+)_(\d+)", name)
        if match:
            comp_type, layer = match.groups()
            return (int(layer), comp_type)
        return (9999, name)

    components = sorted(list(components), key=component_sort_key)
    n = len(components)
    matrix = np.zeros((n, n), dtype=float)
    np.fill_diagonal(matrix, 1.0)
    comp_to_idx = {comp: i for i, comp in enumerate(components)}

    for key, value in pairwise.items():
        parts = key.split("_vs_")
        if len(parts) == 2:
            i, j = comp_to_idx[parts[0]], comp_to_idx[parts[1]]
            matrix[i, j] = value
            matrix[j, i] = value

    return matrix, components

def plot_4x1_top_cbars(base_dir="pipeline/runs", output_name="ortho_4x1_top_cbar.pdf"):
    base_dir = Path(base_dir)
    model_mapping = {
        "Qwen3-4B": "Qwen3-4B",
        "gemma-2-2b-it": "Gemma2-2B-IT",
        "Mistral-Small-3.2-24B-Instruct-2506": "Mistral-3.2-Small",
        "Phi-4-mini-instruct": "Phi4-Mini"
    }

    json_filename = "component_orthogonality_deterministic.json"
    valid_folders = list(model_mapping.keys())

    # Wider figure to handle horizontal colorbars and larger ticks
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    cmap = LinearSegmentedColormap.from_list("red_gray_blue", [(0.0, "#c0392b"), (0.5, "#d9d9d9"), (1.0, "#2c7fb8")])

    for i, folder_name in enumerate(valid_folders):
        json_path = base_dir / folder_name / json_filename
        if not json_path.exists(): continue

        with open(json_path, "r") as f:
            data = json.load(f)

        matrix, components = build_similarity_matrix(data["pairwise_orthogonality"])
        
        # Plot heatmap
        im = sns.heatmap(
            matrix,
            ax=axes[i],
            xticklabels=components, 
            yticklabels=False,
            cmap=cmap,
            center=0,
            square=True,
            cbar=False, # We will create the colorbar manually
            linewidths=0
        )

        # Create divider for the TOP colorbar
        divider = make_axes_locatable(axes[i])
        # Append axes to the top, size is height of cbar, pad is distance from plot
        cax = divider.append_axes("top", size="5%", pad=0.1)
        
        # Add the horizontal colorbar
        cbar = fig.colorbar(
            im.get_children()[0], 
            cax=cax, 
            orientation='horizontal',
            ticklocation='top'
        )
        cbar.ax.tick_params(labelsize=15)

        # Title formatting - positioned above the colorbar
        axes[i].set_title(model_mapping[folder_name], fontsize=20, fontweight='bold', y=1.15)
        
        # Make ticks larger and more legible
        axes[i].tick_params(axis='x', rotation=45, labelsize=12, length=4, width=1)
        
        if i == 0:
            axes[i].set_ylabel("Components (Symmetric)", fontsize=16, fontweight='bold')

    # Pull title into y=0.95
    plt.suptitle(
        "Cosine Similarity between Component Writes Projected onto Refusal Direction", 
        fontsize=24, fontweight='bold', y=1
    )

    # Tight layout with custom rect to prevent title clipping
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_name}")

if __name__ == "__main__":
    plot_4x1_top_cbars()