import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Configuration
base_path = "pipeline/runs"
name_map = {
    "gemma-2-2b-it":                          "Gemma2-2B",
    "Qwen3-4B":                               "Qwen3-4B",
    "Phi-4-mini-instruct":                    "Phi4-Mini",
    "Mistral-Small-3.2-24B-Instruct-2506":    "Mistral-3.2-Small",
}

model_layers = {
    "gemma-2-2b-it": 13,
    "Qwen3-4B": 19,
    "Phi-4-mini-instruct": 12,
    "Mistral-Small-3.2-24B-Instruct-2506": 16 
}

def plot_direction_grid(ax, data, model_name, layer):
    if torch.is_tensor(data):
        data_np = data.detach().cpu().numpy().flatten()
    else:
        data_np = np.array(data).flatten()
    
    # Outlier detection
    q1, q3 = np.percentile(data_np, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 2.5 * iqr
    upper_bound = q3 + 2.5 * iqr
    
    counts, bins = np.histogram(data_np, bins=100)
    
    for i in range(len(counts)):
        if counts[i] == 0: continue
        
        bin_center = (bins[i] + bins[i+1]) / 2
        is_outlier = (bin_center < lower_bound) or (bin_center > upper_bound)
        
        color = '#e74c3c' if is_outlier else '#34495e'
        edge = 'black' if is_outlier else 'none'
        alpha = 1.0 if is_outlier else 0.6
        lw = 1.5 if is_outlier else 0 
        
        ax.bar(bins[i], counts[i], width=bins[i+1]-bins[i], 
               color=color, edgecolor=edge, linewidth=lw, 
               alpha=alpha, align='edge', log=True)

    ax.set_ylim(bottom=0.5) 

    # --- Formatting ---
    ax.set_title(f"{model_name} (Layer {layer})", fontsize=24, fontweight='bold', pad=20)
    ax.set_xlabel("Dimension Value", fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel("Frequency (Log Scale)", fontsize=18, fontweight='bold', labelpad=10)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Statistics Box (Below Axis)
    mu, std = np.mean(data_np), np.std(data_np)
    num_outliers = np.sum((data_np < lower_bound) | (data_np > upper_bound))
    stats_text = (fr"$\mu$: {mu:.1e}  |  $\sigma$: {std:.1e}  |  Outliers: {num_outliers}")
    
    # Position adjusted for 2x2 layout
    ax.text(0.5, -0.32, stats_text, transform=ax.transAxes, 
            ha='center', va='top', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='gray', alpha=0.9))

# --- Execution ---
fig, axes = plt.subplots(2, 2, figsize=(20, 18)) # 2x2 Square Grid
axes_flat = axes.flatten()

for idx, (folder, display_name) in enumerate(name_map.items()):
    ax = axes_flat[idx]
    file_path = os.path.join(base_path, folder, "generate_directions", "refusal_directions.pt")
    
    if not os.path.exists(file_path):
        ax.text(0.5, 0.5, f"Missing: {display_name}", ha='center', va='center', fontsize=20)
        continue
        
    try:
        directions = torch.load(file_path, map_location='cpu', weights_only=True)
        layer_idx = model_layers.get(folder)
        direction_vec = directions[layer_idx]
        plot_direction_grid(ax, direction_vec, display_name, layer_idx)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {display_name}", ha='center', va='center', fontsize=20)

# Legend placed at the top center
legend_elements = [
    Line2D([0], [0], color='#34495e', lw=8, label='Typical'),
    Line2D([0], [0], color='#e74c3c', marker='s', markersize=15, markeredgecolor='black', label='Outlier', linestyle='None')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
           fontsize=18, frameon=True, ncol=2)

# Adjusting rect to prevent overlap between rows
plt.tight_layout(rect=[0, 0.05, 1, 0.95], h_pad=8.0) 
plt.savefig(os.path.join(base_path, "vector_distributions.png"), dpi=300, bbox_inches='tight')
plt.show()