import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Base path
base_path = "pipeline/runs"

# Name mapping
name_map = {
    "gemma-2-2b-it":                          "Gemma2-2B",
    "Qwen3-4B":                               "Qwen3-4B",
    "Phi-4-mini-instruct":                    "Phi4-Mini",
    "Mistral-Small-3.2-24B-Instruct-2506":    "Mistral-3.2-Small",
}

all_model_layer_data = {}
pattern = r'harmful_Full_layer(\d+)_evaluations\.json'

# Data Extraction
for folder_name, display_name in name_map.items():
    sweep_path = os.path.join(base_path, folder_name, "completions", "full_sweep")
    if not os.path.exists(sweep_path): continue
    
    layer_data = {}
    for filename in os.listdir(sweep_path):
        match = re.search(pattern, filename)
        if match:
            layer = int(match.group(1))
            filepath = os.path.join(sweep_path, filename)
            try:
                with open(filepath, 'r') as f:
                    eval_data = json.load(f)
                if 'llamaguard3_success_rate' in eval_data:
                    layer_data[layer] = eval_data['llamaguard3_success_rate']
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    if layer_data:
        all_model_layer_data[display_name] = layer_data

# --- Plotting ---
num_models = len(all_model_layer_data)
if num_models == 0:
    print("No data found.")
else:
    # Larger scale for readability
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 7), sharey=True)
    if num_models == 1: axes = [axes]

    for idx, (display_name, layers) in enumerate(sorted(all_model_layer_data.items())):
        ax = axes[idx]
        
        sorted_layers = sorted(layers.keys())
        asr_values = [layers[l] for l in sorted_layers]
        
        # Identify peak
        max_asr = max(asr_values)
        max_layer = sorted_layers[asr_values.index(max_asr)]
        
        # Use Bar Chart for the "Bar" feel
        bars = ax.bar(sorted_layers, asr_values, color='#3498db', alpha=0.7, width=0.8, edgecolor='#2980b9')
        
        # Highlight the peak bar specifically
        for i, (l, val) in enumerate(zip(sorted_layers, asr_values)):
            if l == max_layer:
                bars[i].set_color('#e74c3c')
                bars[i].set_edgecolor('#c0392b')
                bars[i].set_alpha(0.9)

        # Annotation: "85% (L12)"
        ax.annotate(f'{max_asr:.1%}\n(L{max_layer})', 
                    xy=(max_layer, max_asr), 
                    xytext=(0, 12), 
                    textcoords='offset points', 
                    ha='center', 
                    fontsize=18, 
                    fontweight='bold',
                    color='#c0392b',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#e74c3c', alpha=0.8))

        # Formatting
        ax.set_title(display_name, fontsize=20, fontweight='bold', pad=2)
        ax.set_xlabel('Layer Index', fontsize=14)
        if idx == 0:
            ax.set_ylabel('Attack Success Rate (ASR)', fontsize=16)
        
        # Ticks every 4
        if sorted_layers:
            tick_range = range(min(sorted_layers), max(sorted_layers) + 1, 4)
            ax.set_xticks(list(tick_range))
            ax.tick_params(axis='x', labelsize=16)
            
        ax.set_ylim(0, 1.15) # Room for labels
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.suptitle("Sweep of All Candidate Directions for Refusal Steering Across Models", fontsize = 24)
    plt.tight_layout()
    output_path = os.path.join(base_path, 'layer_sweep_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Final bar graph saved to: {output_path}")
    plt.show()