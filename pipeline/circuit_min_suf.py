import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def component_sort_key(name):
    match = re.match(r"([a-zA-Z]+)_(\d+)", name)
    if match:
        comp_type, layer = match.groups()
        return (int(layer), comp_type)
    return (9999, name)


def load_exhaustive_asr(json_path):
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    sorted_data = sorted(data, key=lambda x: component_sort_key(x["component"]))

    components = [item["component"] for item in sorted_data]
    drop_asrs = [item["drop in ASR"] for item in sorted_data]

    return components, drop_asrs


def plot_minimum_sufficiency(json_path, output_path=None, model_name=None, y_max=None):
    components, drop_asrs = load_exhaustive_asr(json_path)

    # Make plot taller
    plt.figure(figsize=(max(12, len(components) * 0.7), 9))

    plt.plot(range(len(components)), drop_asrs, marker="o", linewidth=2)

    title_name = model_name if model_name is not None else Path(json_path).parent.name
    plt.title(
        f"Circuit Minimum Sufficiency\nModel Used: {title_name}",
        fontsize=16,
        pad=14
    )
    plt.xlabel("Component", fontsize=12)
    plt.ylabel("Drop in ASR", fontsize=12)
    plt.xticks(range(len(components)), components, rotation=45, ha="right")

    # Auto-scale y-axis if not provided
    if y_max is None:
        ymax = max(drop_asrs) * 1.1 if drop_asrs else 1.0
        plt.ylim(0, ymax)
    else:
        plt.ylim(0, y_max)

    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.show()
    plt.close()


def plot_all_models(base_dir="pipeline/runs", y_max=None):
    base_dir = Path(base_dir)

    model_folders = [
        "gemma-2-2b-it",
        "Mistral-Small-3.2-24B-Instruct-2506",
        "Phi-4-mini-instruct",
        "Qwen3-4B",
    ]

    json_filename = "final_exhaustive_ASRs_determinstic.json"

    for model_name in model_folders:
        model_dir = base_dir / model_name
        json_path = model_dir / json_filename

        if not json_path.exists():
            print(f"Skipping {model_name}: file not found -> {json_path}")
            continue

        output_path = model_dir / f"{model_name}_circuit_minimum_sufficiency.png"

        plot_minimum_sufficiency(
            json_path=json_path,
            output_path=output_path,
            model_name=model_name,
            y_max=y_max
        )


if __name__ == "__main__":
    plot_all_models(y_max=None)