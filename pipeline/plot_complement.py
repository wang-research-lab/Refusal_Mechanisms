import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("consolidated_results.csv")

name_map = {
    "gemma-2-2b-it":                          "Gemma2-2B",
    "Qwen3-4B":                               "Qwen3-4B",
    "Phi-4-mini-instruct":                    "Phi4-Mini",
    "Mistral-Small-3.2-24B-Instruct-2506":    "Mistral-3.2-Small",
}
df["model_short"] = df["model"].map(name_map).fillna(df["model"])

colors = plt.cm.Dark2(np.linspace(0, 1, len(df)))

fig, ax = plt.subplots(figsize=(8, 6))

for (_, row), color in zip(df.iterrows(), colors):
    mechanism_x  = row["component_pct"]
    mechanism_y  = row["comp_success_pct"]
    complement_x = 1.0 - row["component_pct"]
    complement_y = row["complement_success_rate"] / row["full_success_rate"]

    # Arrow from complement → mechanism
    ax.annotate(
        "",
        xy=(mechanism_x, mechanism_y),
        xytext=(complement_x, complement_y),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=3,
            mutation_scale=20  # increase this (default ~10)
        ),
    )
    # Mechanism marker: filled circle
    ax.scatter(mechanism_x, mechanism_y, marker="o", s=180,
               color=color, zorder=5)

    # Complement marker: hollow diamond
    ax.scatter(complement_x, complement_y, marker="D", s=180,
               facecolors="none", edgecolors=color, linewidths=2.0, zorder=5)

    # Label at mechanism point
    if "Qwen" in row["model_short"]:
        pos = (-90, 2)
    elif "Phi" in row["model_short"]:
        pos = (12, -3)
    elif "Mistral" in row["model_short"]:
        pos = (8, 0)
    else: 
        pos = (3, 4)
    ax.annotate(
        row["model_short"],
        xy=(mechanism_x, mechanism_y),
        xytext=pos, textcoords="offset points",
        fontsize=15, color=color
    )

ax.set_xlabel("Component Fraction (% of upstream)", fontsize=18)
ax.set_ylabel("ASR as Fraction of Full Steering", fontsize=18)
ax.set_title("Mechanism vs Complement\nComponent Fraction and Steering Effectiveness", fontsize=18)
ax.tick_params(labelsize=11)
ax.set_xlim(0, 1.05)

legend_elements = [
    plt.Line2D([0], [0], marker="o", color="grey", linestyle="none",
               markersize=15, label="Identified mechanism"),
    plt.Line2D([0], [0], marker="D", color="grey", linestyle="none",
               markersize=15, markerfacecolor="none", label="Complement"),
]
ax.legend(handles=legend_elements, fontsize=15, loc="lower left")

plt.tight_layout()
plt.savefig("circuit_vs_complement.png", dpi=150)
plt.close()
print("Saved to circuit_vs_complement.png")