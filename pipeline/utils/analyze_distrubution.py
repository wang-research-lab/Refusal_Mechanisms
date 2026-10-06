# Calculates howmuch of the total steering signal is captured

import torch
import os
import argparse
import numpy as np

def analyze_magnitude_capture(run_dir, top_k=50):
    means_path = os.path.join(run_dir, "generate_directions", "cat_means.pt")
    
    if not os.path.exists(means_path):
        print(f"Error: {means_path} not found.")
        return

    print(f"📊 Analyzing Magnitude Distribution (Top-k={top_k})\n")
    print(f"{'Layer':<10} | {'Total Magnitude':<15} | {'Top-K Magnitude':<15} | {'% Captured':<10}")
    print("-" * 60)

    data = torch.load(means_path, map_location='cpu')
    first_cat = next(iter(data.keys()))
    _, vec_tensor = data[first_cat] # [pos, layers, hidden]

    # Goes through each layer
    num_layers = vec_tensor.shape[1]
    
    low_capture_layers = []

    for layer_idx in range(num_layers):
        # Get the dense vector for this layer
        vec = vec_tensor[-1, layer_idx, :] # [4096]
        
        # Calculate magnitudes (absolute value of activations)
        abs_vec = torch.abs(vec)
        total_mag = torch.sum(abs_vec).item()
        
        # Find the Top-K neurons by magnitude
        top_values, _ = torch.topk(abs_vec, top_k)
        top_k_mag = torch.sum(top_values).item()
        
        # Calculate percentage captured
        if total_mag > 0:
            pct_captured = (top_k_mag / total_mag) * 100
        else:
            pct_captured = 0.0
            
        print(f"{layer_idx:<10} | {total_mag:<15.2f} | {top_k_mag:<15.2f} | {pct_captured:<10.2f}%")

        if pct_captured < 1.0:
            low_capture_layers.append(layer_idx)

    print("-" * 60)
    print("\nCONCLUSION:")
    if len(low_capture_layers) > num_layers / 2:
        print(f"High Uniformity Detected. {len(low_capture_layers)}/{num_layers} layers have <1% signal in Top-{top_k}.")
        print("This confirms Llama-3.1-8B representations are too distributed for sparse steering.")
    else:
        print("Signal appears sufficiently sparse for steering.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to pipeline run")
    parser.add_argument("--top_k", type=int, default=50, help="Number of neurons to keep")
    args = parser.parse_args()

    analyze_magnitude_capture(args.run_dir, args.top_k)