import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Configuration ---
base_path = "pipeline/runs"
name_map = {
    "gemma-2-2b-it":                          "Gemma2-2B",
    "Qwen3-4B":                               "Qwen3-4B",
    "Phi-4-mini-instruct":                    "Phi4-Mini",
    "Mistral-Small-3.2-24B-Instruct-2506":    "Mistral-3.2-Small",
}

colors = {"mlp": "#1f77b4", "attention": "#ff7f0e"} 

def plot_paired_projections(ax, folder_path, model_name, is_first_col):
    paths = {
        "harmful": os.path.join(folder_path, "component_cossim_deterministic_harmful.json"),
        "harmless": os.path.join(folder_path, "component_cossim_deterministic_harmless.json")
    }
    
    if not all(os.path.exists(p) for p in paths.values()):
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', fontsize=16)
        return

    # Load data
    data = {k: json.load(open(p, 'r')) for k, p in paths.items()}
    
    # Extract unique layer integers to anchor the x-axis
    all_keys = data["harmful"].keys()
    layer_indices = sorted(list(set(int(k.split('_')[1]) for k in all_keys)))
    
    for layer_idx in layer_indices:
        # We plot Attn at Layer-0.2 and MLP at Layer+0.2 so they are tangential/grouped
        for comp_type, offset in [("attn", -0.2), ("mlp", 0.2)]:
            # Match naming convention in your JSON
            key = next((k for k in all_keys if f"{comp_type}_{layer_idx}" in k.lower()), None)
            if not key: continue
            
            # This is the clustered x-coordinate
            i = layer_idx + offset
            
            base_color = colors["attention" if "attn" in key.lower() else "mlp"]
            med_h = data["harmful"][key]['median']
            med_hl = data["harmless"][key]['median']

            # 1. Harmful Box (Wider, desaturated base color)
            h_stats = data["harmful"][key]
            h_item = [{'whislo': h_stats['q25'], 'q1': h_stats['q25'], 'med': h_stats['median'], 
                       'q3': h_stats['q75'], 'whishi': h_stats['q75'], 'fliers': []}]
            
            box_h = ax.bxp(h_item, positions=[i], showfliers=False, widths=0.35, 
                           patch_artist=True, showcaps=False, zorder=2)
            for patch in box_h['boxes']:
                patch.set(facecolor=base_color, alpha=0.4, edgecolor=base_color, linewidth=1)
            for med in box_h['medians']:
                med.set(color='white', linewidth=1, zorder=3)

            # 2. Harmless Box (Narrower, high-contrast gray)
            hl_stats = data["harmless"][key]
            hl_item = [{'whislo': hl_stats['q25'], 'q1': hl_stats['q25'], 'med': hl_stats['median'], 
                        'q3': hl_stats['q75'], 'whishi': hl_stats['q75'], 'fliers': []}]
            
            box_hl = ax.bxp(hl_item, positions=[i], showfliers=False, widths=0.5, 
                            patch_artist=True, showcaps=False, zorder=1)
            for patch in box_hl['boxes']:
                patch.set(facecolor='#e0e0e0', alpha=0.9, edgecolor='black', linewidth=0.6)
            for med in box_hl['medians']:
                med.set(color='black', linewidth=1, zorder=5)

            # 3. Displacement Arrow (Top layer, bold red)
            ax.annotate('', xy=(i, med_hl), xytext=(i, med_h),
                        arrowprops=dict(arrowstyle='-|>', color='red', lw=1.5, 
                                        mutation_scale=10, zorder=10, shrinkA=0, shrinkB=0))

    # --- Formatting ---
    ax.set_title(model_name, fontsize=24, fontweight='bold', pad=20)
    ax.axhline(0, color='black', lw=1.2, alpha=0.5)
    
    # Strict Y-axis limits for CosSim detail
    ax.set_ylim(-0.2, 0.3) 
    
    # Set ticks on the actual layer integers
    ax.set_xticks(layer_indices)
    # Label every 2nd layer for clarity
    labels = [str(l) if l % 2 == 0 else "" for l in layer_indices]
    ax.set_xticklabels(labels, fontsize=12)
    
    if is_first_col:
        ax.set_ylabel("Cosine Similarity", fontsize=20, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
    else:
        ax.tick_params(axis='y', labelleft=False, left=False)
    
    ax.set_xlabel("Layer Index", fontsize=16)
    ax.grid(True, axis='y', ls=":", alpha=0.3)

# --- Main Execution ---
# 1. Initialize 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(24, 16)) # Adjusted aspect ratio for 2x2
axes_flat = axes.flatten()

for idx, (folder, display_name) in enumerate(name_map.items()):
    full_path = os.path.join(base_path, folder)
    # Determine if it's in the first column (index 0 or 2)
    is_first_col = (idx % 2 == 0)
    
    # Pass the flattened axis to the plotting function
    plot_paired_projections(axes_flat[idx], full_path, display_name, is_first_col)

# 2. Refined Layout for 2x2
# Increased hspace for title/label breathing room
plt.subplots_adjust(top=0.90, wspace=0.05, hspace=0.25, bottom=0.15, left=0.08, right=0.95)

# 3. Global Legend
legend_elements = [
    Line2D([0], [0], color='w', marker='s', markersize=14, markerfacecolor=colors['mlp'], label='MLP (Harmful)', alpha=0.5),
    Line2D([0], [0], color='w', marker='s', markersize=14, markerfacecolor=colors['attention'], label='Attention (Harmful)', alpha=0.5),
    Line2D([0], [0], color='w', marker='s', markersize=10, markerfacecolor='#e0e0e0', markeredgecolor='black', label='Harmless IQR'),
    Line2D([0], [0], color='red', lw=2.5, marker='>', label='Selectivity Shift (Harmful $\\rightarrow$ Harmless)')
]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=18, frameon=False)

plt.suptitle("Component Refusal Displacement: Harmful vs. Harmless Instructions", fontsize=34, fontweight='bold', y=0.96)

# Save and Show
save_path = os.path.join(base_path, "paired_projection_displacement.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()