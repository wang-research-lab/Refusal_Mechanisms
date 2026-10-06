import torch
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

# ── Configuration ────────────────────────────────────────────────────────────
base_path = "pipeline/runs"
name_map = {
    "gemma-2-2b-it":                          "Gemma 2-2B",
    "Qwen3-4B":                               "Qwen3-4B",
    "Phi-4-mini-instruct":                    "Phi4-Mini",
    "Mistral-Small-3.2-24B-Instruct-2506":    "Mistral 3.2-Small",
}

CHOSEN_LAYERS = {
    "gemma-2-2b-it": 13,
    "Qwen3-4B": 19,
    "Phi-4-mini-instruct": 14,
    "Mistral-Small-3.2-24B-Instruct-2506": 17,
}

PALETTE = {
    "attention": "#E8734A",   # warm terracotta
    "mlp":       "#4A90C4",   # cool steel blue
    "zero":      "#D0D0D0",   # light grey placeholder
}

# ── Data loading ─────────────────────────────────────────────────────────────
def get_fixed_direction_contributions(model_folder):
    dir_path = os.path.join(base_path, model_folder, "generate_directions", "refusal_directions.pt")
    h_path   = os.path.join(base_path, model_folder, "component_mean_output_norms_deterministic_harmful.pt")
    hl_path  = os.path.join(base_path, model_folder, "component_mean_output_norms_deterministic_harmless.pt")

    if not all(os.path.exists(p) for p in [dir_path, h_path, hl_path]):
        return None

    refusal_dirs_dict = torch.load(dir_path)
    means_h  = torch.load(h_path)
    means_hl = torch.load(hl_path)

    chosen_idx = CHOSEN_LAYERS.get(model_folder)
    if chosen_idx not in refusal_dirs_dict:
        chosen_idx = sorted(refusal_dirs_dict.keys())[len(refusal_dirs_dict) // 2]

    r_vec = refusal_dirs_dict[chosen_idx]
    r_hat = (r_vec / (r_vec.norm() + 1e-8)).detach().cpu().float()

    contributions = {}
    for comp_name in means_h:
        if comp_name in means_hl:
            diff = (means_h[comp_name] - means_hl[comp_name]).detach().cpu().float()
            contributions[comp_name] = torch.dot(diff, r_hat).item()

    return contributions


# ── Layout builder ────────────────────────────────────────────────────────────
def build_interleaved_layout(contributions):
    """
    Fixes the interleaved pattern by using absolute layer indices.
    This ensures that Layer N's Attention is always at 2N and MLP is at 2N+1.
    """
    # 1. Extract all layer numbers to find the range
    layer_nums = []
    for k in contributions:
        parts = k.split("_")
        for p in parts:
            if p.isdigit():
                layer_nums.append(int(p))
                break
    
    if not layer_nums:
        return np.array([]), np.array([]), [], []

    min_l, max_l = min(layer_nums), max(layer_nums)
    all_layers = np.arange(min_l, max_l + 1)
    
    # 2. Create slots for EVERY layer in the range (2 slots per layer)
    # This prevents "shifting" if a specific layer is missing from the data
    n_layers = len(all_layers)
    positions = np.arange(n_layers * 2, dtype=float)
    values = np.zeros(n_layers * 2)
    bar_colors = [PALETTE["zero"]] * (n_layers * 2)

    for k, v in contributions.items():
        layer_num = None
        for p in k.split("_"):
            if p.isdigit():
                layer_num = int(p)
                break
        
        if layer_num is None: continue
        
        is_attn = "attn" in k.lower()
        # Offset by min_l so the first bar starts at index 0
        relative_layer = layer_num - min_l
        slot = relative_layer * 2 + (0 if is_attn else 1)
        
        values[slot] = v
        bar_colors[slot] = PALETTE["attention"] if is_attn else PALETTE["mlp"]

    # 3. Create ticks centered between the Attn and MLP bars of each layer
    layer_ticks = []
    for i, l_val in enumerate(all_layers):
        center_pos = i * 2 + 0.5
        layer_ticks.append((center_pos, str(l_val)))

    return positions, values, bar_colors, layer_ticks


# ── Plotting ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Georgia", "DejaVu Serif", "Times New Roman"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  True,
    "axes.spines.bottom":False,
})

fig, axes = plt.subplots(1, 4, figsize=(24, 9), sharey=True)
fig.patch.set_facecolor("#FAFAF8")

for idx, (folder, display_name) in enumerate(name_map.items()):
    ax = axes[idx]

    data = get_fixed_direction_contributions(folder)

    if data is None:
        ax.text(0.5, 0.5, "Data Missing", ha="center", va="center",
                fontsize=14, color="#999", transform=ax.transAxes)
        continue

    positions, values, bar_colors, layer_ticks = build_interleaved_layout(data)
    total = float(np.sum(values))

    # ── Bars ──────────────────────────────────────────────────────────────────
    bars = ax.bar(
        positions, values, color=bar_colors,
        width=0.72, edgecolor="white", linewidth=0.4,
        zorder=3,
    )

    # Make zero-placeholder bars nearly invisible
    for bar, col in zip(bars, bar_colors):
        if col == PALETTE["zero"]:
            bar.set_alpha(0.0)   # fully hide absent components

    # ── Zero line ─────────────────────────────────────────────────────────────
    ax.axhline(0, color="#888", lw=0.8, zorder=2)

    # ── Reconstruction annotation ─────────────────────────────────────────────
    chosen_l = CHOSEN_LAYERS.get(folder, "?")
    sign = "+" if total >= 0 else "−"
    abs_total = abs(total)
    recon_text = f"Σ = {sign}{abs_total:.3f}"

    ax.text(
        0.1, 0.97, recon_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=18,
        fontweight="bold",
        color="#222",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#CCCCCC",
            linewidth=0.8,
            alpha=0.9,
        ),
        zorder=5,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"{display_name}",
        fontsize=18, fontweight="bold", pad=10, color="#111",
    )

    # ── Y label (leftmost only) ───────────────────────────────────────────────
    if idx == 0:
        ax.set_ylabel(
            "Contribution to Refusal Direction",
            fontsize=24, labelpad=10, color="#333",
        )
    ax.tick_params(axis="y", labelsize=14, colors="#444")

    # ── X-axis: one tick per layer, centred between attn & mlp slots ─────────
    tick_positions = [pos for pos, _ in layer_ticks]
    tick_labels    = [lbl for _, lbl in layer_ticks]

    # Show every other label to avoid crowding
    stride = max(1, len(tick_labels) // 14)
    sparse_positions = tick_positions[::stride]
    sparse_labels    = tick_labels[::stride]

    ax.set_xticks(sparse_positions)
    ax.set_xticklabels(sparse_labels, fontsize=14, color="#555")
    ax.set_xlabel("Layer", fontsize=16, color="#555", labelpad=6)

    # Subtle y-grid
    ax.yaxis.grid(True, linestyle=":", linewidth=1, color="#CCCCCC", alpha=0.7, zorder=0)
    ax.xaxis.grid(True, linestyle=":", linewidth=1, color="#CCCCCC", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#333333")
    ax.spines["left"].set_linewidth(2)

    # tight x margins
    ax.set_xlim(-1, positions[-1] + 1)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=PALETTE["attention"], edgecolor="white", label="Attention head"),
    mpatches.Patch(facecolor=PALETTE["mlp"],       edgecolor="white", label="MLP layer"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=2,
    fontsize=18,
    frameon=False,
    handlelength=1.4,
    handleheight=1.0,
)

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Linear Attribution of Component Writes to the Refusal Direction",
    fontsize=26, fontweight="bold", y=1.02, color="#111",
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
out_path = os.path.join(base_path, "fixed_layer_attribution_bars.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
plt.show()