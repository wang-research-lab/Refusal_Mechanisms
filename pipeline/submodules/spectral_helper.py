import torch
import numpy as np
from typing import Dict, Tuple, Optional
import json
import os
from tqdm import tqdm


def analyze_direction_eigenspace_structure(
    direction: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    top_k_coords: np.ndarray,
    bot_k_coords: np.ndarray,
) -> Dict:
    d = direction.float()
    d_norm = d / d.norm()
    
    def get_spectral_stats(d_sparse):
        d_sparse_norm = d_sparse / (d_sparse.norm() + 1e-8)
        coords = eigenvectors.T @ d_sparse_norm
        weights = coords ** 2
        weights = weights / (weights.sum() + 1e-8)
        pr = (1.0 / (weights ** 2).sum()).item()
        entropy = -(weights * (weights + 1e-10).log()).sum().item()
        top10_mass = weights[:10].sum().item()
        top50_mass = weights[:50].sum().item()
        top100_mass = weights[:100].sum().item()
        return {
            'participation_ratio': pr,
            'entropy': entropy,
            'top10_mass': top10_mass,
            'top50_mass': top50_mass,
            'top100_mass': top100_mass,
        }

    # Full direction
    full_stats = get_spectral_stats(d_norm)

    # Top magnitude coordinates
    d_top = torch.zeros_like(d)
    d_top[top_k_coords] = d[top_k_coords]
    top_stats = get_spectral_stats(d_top)

    # Bottom magnitude coordinates
    d_bot = torch.zeros_like(d)
    d_bot[bot_k_coords] = d[bot_k_coords]
    bot_stats = get_spectral_stats(d_bot)

    return {
        'full': full_stats,
        'top_mag': top_stats,
        'bot_mag': bot_stats,
        'eigenvalues_top20': eigenvalues[:20].cpu().numpy().tolist(),
    }


def run_eigenspace_analysis(
    model_base,
    nontarget_directions: Dict[int, torch.Tensor],
    layers_to_analyze: list,
    top_budget: int,
    bot_budget: int,
    output_path: str,
):
    results = {}

    for layer_idx in tqdm(layers_to_analyze, desc="Analyzing eigenspace structure"):
        print(f"\n>>> Layer {layer_idx}")

        # F_local uses layer_idx for the weight matrices
        # but DIM direction is computed at layer input, so layer_idx-1
        eigenvalues, eigenvectors, F_local = model_base.compute_f_local(layer_idx)

        direction = nontarget_directions[layer_idx - 1].float().to(eigenvectors.device)

        coord_magnitudes = direction.abs().cpu().numpy()
        sorted_indices = np.argsort(coord_magnitudes)[::-1]
        top_mag_indices = sorted_indices[:top_budget].copy()
        bot_mag_indices = sorted_indices[-bot_budget:].copy()

        layer_results = analyze_direction_eigenspace_structure(
            direction, eigenvalues, eigenvectors,
            top_k_coords=top_mag_indices,
            bot_k_coords=bot_mag_indices,
        )

        layer_results['layer'] = layer_idx
        layer_results['top_budget'] = top_budget
        layer_results['bot_budget'] = bot_budget
        layer_results['hidden_size'] = direction.shape[0]

        results[str(layer_idx)] = layer_results

        print(f"  Full   - PR: {layer_results['full']['participation_ratio']:.1f}, "
              f"top10 mass: {layer_results['full']['top10_mass']:.3f}")
        print(f"  TopMag - PR: {layer_results['top_mag']['participation_ratio']:.1f}, "
              f"top10 mass: {layer_results['top_mag']['top10_mass']:.3f}")
        print(f"  BotMag - PR: {layer_results['bot_mag']['participation_ratio']:.1f}, "
              f"top10 mass: {layer_results['bot_mag']['top10_mass']:.3f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    return results

def measure_eigenspace_overlap(
    eigenvectors_a: torch.Tensor,
    eigenvectors_b: torch.Tensor,
    top_k: int = 50,
) -> Dict:
    V_a = eigenvectors_a[:, :top_k]
    V_b = eigenvectors_b[:, :top_k]

    cross_gram = V_a.T @ V_b
    singular_values = torch.linalg.svdvals(cross_gram).clamp(0, 1)
    principal_angles = torch.acos(singular_values)

    return {
        'grassmann_distance': principal_angles.norm().item(),
        'mean_principal_cos': singular_values.mean().item(),
        'effective_overlap_dim': (principal_angles < 0.1).sum().item(),
    }