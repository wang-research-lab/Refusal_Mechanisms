import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker

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
    "Phi-4-mini-instruct": 14,
    "Mistral-Small-3.2-24B-Instruct-2506": 17
}

def get_pro_ink_cmap(cmap_name='inferno'):
    base_cmap = plt.get_cmap(cmap_name)
    colors = base_cmap(np.linspace(0, 0.85, 256))
    return ListedColormap(colors)

def plot_pro_sorted(axes_pair, data, refusal_vec, model_name, is_first_col):
    ax_top, ax_bot = axes_pair
    ink_cmap = get_pro_ink_cmap('inferno')
    
    # 1. Sort by Refusal Direction Magnitude
    ref_np = refusal_vec.detach().cpu().numpy().flatten()
    sort_indices = np.argsort(np.abs(ref_np))[::-1]
    
    # 2. Filter Top 10%
    top_k = int(len(sort_indices) * 0.1)
    keep_indices = sort_indices[:top_k]
    x_axis = np.arange(top_k)
    
    
    ax_top.set_xlim(-5, top_k + 5)
    ax_bot.set_xlim(-5, top_k + 5)

    sorted_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
    max_layer = max([int(k.split('_')[1]) for k in data.keys()])

    # --- Refusal Vector Baseline (The "Expected" Distribution) ---
    # Unit-normalize magnitudes for a fair comparison
    ref_vals = np.abs(ref_np[sort_indices[:top_k]])
    ref_baseline = ref_vals / np.linalg.norm(ref_np)

    for ax in [ax_top, ax_bot]:
        # Shaded region showing the 'ground truth' refusal importance
        ax.fill_between(x_axis, -ref_baseline, ref_baseline, color='gray', alpha=0.15, label='_nolegend_')
        ax.plot(x_axis, ref_baseline, color='gray', linewidth=1, alpha=0.3, label='_nolegend_')
        ax.plot(x_axis, -ref_baseline, color='gray', linewidth=1, alpha=0.3, label='_nolegend_')

    for key in sorted_keys:
        comp_type, layer_idx = key.split('_')
        layer_idx = int(layer_idx)
        
        v = data[key].float()
        norm = torch.norm(v)
        unit_val = (v / norm).detach().cpu().numpy().flatten() if norm > 0 else v.numpy().flatten()
        
        sorted_vals = unit_val[keep_indices]
        color = ink_cmap(layer_idx / max_layer)
        target_ax = ax_top if comp_type == "mlp" else ax_bot
        
        point_sizes = 2 + (np.abs(sorted_vals) * 1250)
        
        target_ax.scatter(x_axis, sorted_vals, 
                          s=point_sizes, 
                          color=color, 
                          alpha=0.6, 
                          marker='o', 
                          edgecolors='none', 
                          rasterized=True)

    # --- Formatting ---
    for ax, label in zip([ax_top, ax_bot], ["MLP", "Attn"]):
        ax.set_ylim(-0.16, 0.16) 
        ax.axhline(y=0, color='black', linewidth=1.0, alpha=0.5)
        
        if is_first_col:
            ax.set_ylabel(label, fontsize=20, fontweight='bold')
            # Hard-override the visibility
            ax.yaxis.set_major_locator(ticker.FixedLocator([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]))
            ax.yaxis.set_major_formatter(ticker.FixedFormatter(['-0.15', '-0.10', '-0.05', '0', '0.05', '0.10', '0.15']))
            ax.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=18)
        else:
            ax.yaxis.set_tick_params(labelleft=False, left=False)

        ax.grid(axis='y', linestyle=':', alpha=0.4)

    ax_top.set_title(f"{model_name}", fontsize=24, fontweight='bold', pad=20)
    ax_top.set_xticks([]) 
    ax_bot.set_xlabel("Refusal Magnitude Rank\n(Descending)", fontsize=16, fontweight='bold')

# --- Execution ---
fig, axes = plt.subplots(2, 4, figsize=(32, 14), sharex='col', sharey='row')
pro_cmap = get_pro_ink_cmap('inferno')

for idx, (folder, display_name) in enumerate(name_map.items()):
    axes_pair = [axes[0, idx], axes[1, idx]]
    comp_path = os.path.join(base_path, folder, "component_mean_output_norms_deterministic_harmful.pt")
    vec_path = os.path.join(base_path, folder, "generate_directions", "refusal_directions.pt")
    
    is_first_col = (idx == 0)
    
    if os.path.exists(comp_path) and os.path.exists(vec_path):
        ref_vec = torch.load(vec_path, map_location='cpu', weights_only=True)[model_layers[folder]]
        comp_data = torch.load(comp_path, map_location='cpu', weights_only=True)
        plot_pro_sorted(axes_pair, comp_data, ref_vec, display_name, is_first_col)

# Final Layout Adjustments - Added left margin so the numbers aren't cut off
plt.subplots_adjust(top=0.88, hspace=0.0, wspace=0.01, bottom=0.01, left=0.08)

# Colorbar
cbar_ax = fig.add_axes([0.3, -0.0, 0.4, 0.015])
sm = plt.cm.ScalarMappable(cmap=pro_cmap, norm=plt.Normalize(vmin=0, vmax=1))
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_ticks([]) # Remove the numeric ticks
cbar.set_label('Layer Depth (Early $\\rightarrow$ Late)', fontsize=21, fontweight='bold')

# --- Global Legend and Final Save ---
from matplotlib.lines import Line2D

# Create proxy elements for the legend
legend_elements = [
    Line2D([0], [0], color='gray', lw=12, alpha=0.6, label='Abs. Refusal Vector Envelope ($|\\mathbf{r}^*_i|$)'),
    Line2D([0], [0], marker='o', color='w', label='Component Write (Unit-Normalized)',
           markerfacecolor='black', markersize=12, alpha=0.6),
    Line2D([0], [0], color='white', label='Point Size $\\propto$ Write Magnitude')
]

# Adjust layout to make room for a legend at the bottom
# Increased the bottom margin from 0.01 to 0.12
plt.subplots_adjust(top=0.88, hspace=0.0, wspace=0.01, bottom=0.12, left=0.08)

# Add the legend to the figure
fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
           fontsize=20, frameon=True, bbox_to_anchor=(0.5, 0.02))

plt.savefig(os.path.join(base_path, "component_output_distributions.png"), dpi=300, bbox_inches='tight')
plt.show()

# Suptitle
plt.suptitle("Component Write Profiles vs. Refusal Direction Dimension Structure", 
             fontsize=34, fontweight='bold', y=0.97)

# Save with tight_layout turned off to respect our custom add_axes
plt.savefig(os.path.join(base_path, "component_output_distributions.png"), dpi=300, bbox_inches='tight')
plt.show()