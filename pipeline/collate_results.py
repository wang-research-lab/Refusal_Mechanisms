import pandas as pd
import json
import re
from pathlib import Path
import numpy as np

def consolidate_evaluations(base_path="pipeline/runs"):
    base_dir = Path(base_path)
    all_data = []

    # Regex to find the layer number in the anchor file
    anchor_pattern = re.compile(r"exhaustive_set_anchor_(\d+)_determinstic\.json")
    coords_pattern = re.compile(r"(\d+)coords")
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # 1. Dynamically find the anchor file and extract the layer
        anchor_file = None
        layer_num = None
        
        for file in model_dir.glob("exhaustive_set_anchor_*_determinstic.json"):
            match = anchor_pattern.search(file.name)
            if match:
                anchor_file = file
                layer_num = match.group(1)
                break
        
        if not anchor_file or not layer_num:
            print(f"Skipping {model_name}: No valid anchor file found.")
            continue

        # 2. Load components and anchor data
        with open(anchor_file, 'r') as f:
            anchor_data = json.load(f)
            components = anchor_data.get("components", [])
            attn_components = [int(c.replace("attn_", "")) for c in components if "attn" in c]
            mlp_components = [int(c.replace("mlp_", "")) for c in components if "mlp" in c]

        # 3. Define directories
        subcomp_dir = model_dir / "completions" / "subcomponent_test"
        full_sweep_dir = model_dir / "completions" / "full_sweep"

        # 4. Helper to extract metrics
        def get_metric(path):
            if path and path.exists():
                with open(path, 'r') as f:
                    return json.load(f).get("llamaguard3_success_rate")
            return None

        # 5. Handle the sparse file via glob (since eXX and XXXXcoords are variable)
        sparse_file_path = None
        if subcomp_dir.exists():
            sparse_matches = list(subcomp_dir.glob("*sparse*evaluations.json"))
            if sparse_matches:
                sparse_file_path = sparse_matches[0] # Take the first match
                coords_match = coords_pattern.search(sparse_file_path.name)
                if coords_match:
                    sparse_coords = int(coords_match.group(1))

        hidden_size = 0
        if "gemma" in model_name:
            hidden_size = 2304
        if "Phi" in model_name:
            hidden_size = 3072
        if "Qwen" in model_name:
            hidden_size = 2560
        if "Mistral" in model_name:
            hidden_size = 5120

        # 6. Standard filenames
        full_eval_path = full_sweep_dir / f"harmful_Full_layer{layer_num}_evaluations.json"
        comp_eval_path = subcomp_dir / f"harmful_SubCompTest_L{layer_num}__deterministic_evaluations.json"
        complement_eval_path = subcomp_dir / f"harmful_SubCompTest_L{layer_num}__deterministic_complement_evaluations.json"

        model_layer_mapping = {
            "Qwen3-4B": 19,
            "gemma-2-2b-it": 13,
            "Mistral-Small-3.2-24B-Instruct-2506": 17,
            "Phi-4-mini-instruct": 14
        }
        # 7. Collect Data
        all_data.append({
            "model": model_name,
            "layer": layer_num,
            "component_pct": np.round((len(attn_components) + len(mlp_components))/(2 * model_layer_mapping[model_name] + 1), 5),
            "attn_components": attn_components,
            "mlp_components": mlp_components,            
            "comp_success_pct": np.round(get_metric(comp_eval_path)/get_metric(full_eval_path),5),
            "sparse_success_pct": np.round(get_metric(sparse_file_path)/get_metric(full_eval_path),5),
            "sparse_coords_pct": np.round(sparse_coords/hidden_size, 5),
            "full_success_rate": get_metric(full_eval_path),
            "complement_success_rate": get_metric(complement_eval_path),
        })

    return pd.DataFrame(all_data)

df = consolidate_evaluations()
df.to_csv("consolidated_results.csv", index=False)
print(df)