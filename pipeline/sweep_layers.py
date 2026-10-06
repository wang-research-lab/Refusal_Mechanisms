import os
import torch
import numpy as np
import json
import argparse
import gc
from scipy.stats import spearmanr
from tqdm import tqdm

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions_repit import generate_category_directions
from pipeline.submodules.completions_helper import generate_and_save_completions_for_dataset, evaluate_completions_and_save_results_for_dataset
from pipeline.submodules.evaluate_jailbreak import unload_llamaguard_model
from pipeline.utils.hook_utils import get_linear_direction_ablation_input_pre_hook, get_linear_direction_ablation_output_hook
from dataset.load_dataset import load_dataset_split


# ── Directions ─────────────────────────────────────────────────────────────────

def generate_and_save_category_directions(cfg, model_base, harmful_train, harmless_train,
                                           batch_size=128, use_existing=False):
    dir_path = os.path.join(cfg.artifact_path(), 'generate_directions')
    os.makedirs(dir_path, exist_ok=True)
    cat_means_path    = os.path.join(dir_path, 'cat_means.pt')
    harmless_ref_path = os.path.join(dir_path, 'harmless_reference.pt')
    if use_existing and os.path.exists(cat_means_path) and os.path.exists(harmless_ref_path):
        return torch.load(cat_means_path), torch.load(harmless_ref_path)
    cat_data, harmless_mean = generate_category_directions(
        model_base, harmful_train, harmless_train, artifact_dir=dir_path, batch_size=batch_size
    )
    torch.save(cat_data, cat_means_path)
    torch.save(harmless_mean, harmless_ref_path)
    return cat_data, harmless_mean


def get_norm_explained_mask(direction, energy_threshold=0.9):
    """
    Return indices of the fewest coordinates that explain `energy_threshold`
    fraction of the direction's squared L2 norm.
    """
    sorted_indices = direction.abs().argsort(descending=True)
    squared        = direction[sorted_indices] ** 2
    cumulative     = squared.cumsum(0) / squared.sum()
    k              = int((cumulative < energy_threshold).sum().item()) + 1
    return sorted_indices[:k]
    
def build_directions(category_directions, num_layers, pos=-1):
    """Average per-category directions into a single mean refusal direction per layer boundary."""
    return {
        layer_idx: torch.mean(
            torch.stack([n * direction[pos, layer_idx]
                         for cat, (n, direction) in category_directions.items()], dim=0), dim=0
        )
        for layer_idx in range(1, num_layers)
    }

def unload_model(model_base):
    import ctypes
    model_base.model     = None
    model_base.tokenizer = None
    del model_base
    gc.collect()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ctypes.CDLL('libc.so.6').malloc_trim(0)

def evaluate_tags(cfg, tags_and_paths, batch_size, use_existing):
    print("\nEvaluating...")
    for tag, save_path in tags_and_paths:
        print(f"  {tag}")
        evaluate_completions_and_save_results_for_dataset(
            cfg, tag, "harmful", eval_methodologies=['llamaguard3'],
            save_path=save_path, use_existing=use_existing, batch_size=batch_size
        )
    unload_llamaguard_model()
    gc.collect()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    import ctypes; ctypes.CDLL('libc.so.6').malloc_trim(0)

# ── Hooks ──────────────────────────────────────────────────────────────────────

def make_sparse_projection_input_hook(direction, mask_indices):
    """
    Linear orthogonalization of the refusal direction, restricted to a sparse
    coordinate mask. Coordinates outside the mask are not projected.
    """
    def hook_fn(module, input):
        activation = input[0] if isinstance(input, tuple) else input
        dtype, device = activation.dtype, activation.device
        d = direction.to(device=device, dtype=dtype)
        norm_d   = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        proj_v   = (activation @ norm_d).unsqueeze(-1) * norm_d
        mask = torch.zeros(d.shape[0], dtype=torch.bool, device=device)
        mask[mask_indices] = True
        proj_v[..., ~mask]   = 0.0
        activation_mod = activation - proj_v
        return (activation_mod, *input[1:]) if isinstance(input, tuple) else activation_mod
    return hook_fn


# ── Random Orthogonal Rotation ─────────────────────────────────────────────────

def sample_haar_rotation(d, dtype=torch.float32, device='cpu', seed=None):
    """
    Sample a Haar-distributed random orthogonal matrix Q ∈ O(d) via QR
    decomposition of a standard Gaussian matrix.

    The sign correction on Q ensures uniform (Haar) measure rather than just
    the QR factor, which is only Haar up to column sign.

    Args:
        d:      dimensionality
        dtype:  torch dtype for Q
        device: torch device
        seed:   optional integer for reproducibility

    Returns:
        Q: (d, d) orthogonal tensor
    """
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        Z = torch.randn(d, d, dtype=torch.float64, device=device, generator=gen)
    else:
        Z = torch.randn(d, d, dtype=torch.float64, device=device)

    Q, R = torch.linalg.qr(Z)
    # Correct sign so the decomposition is Haar-uniform
    signs = R.diagonal().sign()
    Q = Q * signs.unsqueeze(0)
    return Q.to(dtype)


def make_rotated_sparse_projection_hook(direction, mask_indices, Q):
    """
    Null-ablation hook operating in the rotated basis Qr*.

    The masking is applied to the *rotated* direction coordinates — i.e. we
    take the top-k dimensions of r_rot = Q @ r* by magnitude, project only
    those, then rotate back.  Activations themselves are NOT rotated (the
    "coordinate-only" variant described in the experiment spec), so this tests
    whether the *basis alignment* of the original mask matters.

    Concretely, for an activation h:
        r_rot  = Q @ r*
        r_rot_masked[i] = r_rot[i] if i in mask_indices else 0
        proj   = (h · r_rot_masked / ||r_rot_masked||²) * r_rot_masked
        h_out  = h - proj

    Args:
        direction:    original refusal direction r* (d,)
        mask_indices: indices of the top-k coordinates of ||r_rot||
        Q:            (d, d) orthogonal rotation matrix
    """
    def hook_fn(module, input):
        activation = input[0] if isinstance(input, tuple) else input
        dtype, device = activation.dtype, activation.device

        d_orig = direction.to(device=device, dtype=dtype)          # r*,  (d,)
        Q_dev  = Q.to(device=device, dtype=dtype)                   # (d, d)

        r_rot = Q_dev @ d_orig                                       # rotated direction, (d,)

        # Build sparse version of the rotated direction
        r_rot_sparse = torch.zeros_like(r_rot)
        r_rot_sparse[mask_indices] = r_rot[mask_indices]

        # Project activations onto the sparse rotated direction
        norm_sq = (r_rot_sparse @ r_rot_sparse).clamp(min=1e-12)
        coeff   = (activation @ r_rot_sparse) / norm_sq             # (...,)
        proj    = coeff.unsqueeze(-1) * r_rot_sparse                 # (..., d)

        activation_mod = activation - proj
        return (activation_mod, *input[1:]) if isinstance(input, tuple) else activation_mod

    return hook_fn


# ── 1. Norm-explained sparsity sweep ──────────────────────────────────────────
# Refusal directions are sparse: a small fraction of coordinates explains most
# of the squared norm and carries the steering signal. We characterize this by:
#   a) Full-direction steering across all layers to find top performers
#   b) Norm-explained threshold sweep only on those top layers

NORM_SWEEP_THRESHOLDS_TOP = [0.5, 0.7, 0.8, 0.9, 0.95, 0.975]
NORM_SWEEP_THRESHOLDS_BOT = [0.9, 0.95]

# Number of independent rotation trials for the rotation control experiment
NUM_ROTATION_TRIALS = 5


def run_full_sweep(cfg, model_base, directions, harmless_mean,
                   all_layers, artifact_path, model_name,
                   harmful_test, batch_size, use_existing):
    """Full direction steering across all layers — establishes which layers work."""
    tags = []

    layers = [l for l in range(14, 25) if l in directions]
    X = torch.stack([directions[l] for l in layers], dim=0)  
    # shape: [num_layers, d]
    # Center across layers
    X_centered = X - X.mean(dim=0, keepdim=True)
    # SVD
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    # X = U S V^T
    # Principal components in feature space
    components = Vh  # shape: [num_layers, d]

    for layer_idx in all_layers:
        direction      = directions[layer_idx]
        reference      = harmless_mean[-1, layer_idx]
        sorted_indices = direction.abs().argsort(descending=True).cpu().numpy()
        hook_fn = make_sparse_projection_input_hook(direction, sorted_indices.copy())
        tag     = f'Full_layer{layer_idx}'
        tags.append((tag, "completions/full_sweep"))
        
        generate_and_save_completions_for_dataset(
            cfg, model_base, [(model_base.model_block_modules[layer_idx], hook_fn)], [],
            tag, 'harmful', batch_size=batch_size, dataset=harmful_test,
            save_path="completions/full_sweep", use_existing=use_existing
        )
        

    return tags


def get_top_layers_by_asr(cfg, all_layers, top_k=1):
    """Rank layers by full-direction ASR and return top-k."""
    results = []
    for layer_idx in all_layers:
        path = os.path.join(cfg.artifact_path(),
                            f"completions/full_sweep/harmful_Full_layer{layer_idx}_evaluations.json")
        if os.path.exists(path):
            with open(path) as f:
                asr = json.load(f)['llamaguard3_success_rate']
            results.append((layer_idx, asr))
    results.sort(key=lambda x: x[1], reverse=True)
    top = sorted([l for l, _ in results[:top_k]])
    print(f"  Top-{top_k} layers by ASR: {top}")
    return top


def run_norm_sweep(cfg, model_base, directions, harmless_mean,
                   sweep_layers, artifact_path, model_name,
                   harmful_test, batch_size, use_existing):
    """
    Norm-explained threshold sweep on representative layers.
    Top: minimum coordinates explaining t fraction of squared norm.
    Bottom: complement at conservative thresholds only.
    """
    tags = []
    for layer_idx in sweep_layers:
        direction      = directions[layer_idx]
        reference      = harmless_mean[-1, layer_idx]
        hidden_size    = direction.shape[0]
        sorted_indices = direction.abs().argsort(descending=True).cpu().numpy()

        for t in NORM_SWEEP_THRESHOLDS_TOP:
            mask    = get_norm_explained_mask(direction, t)
            k       = len(mask)
            hook_fn = make_sparse_projection_input_hook(direction, mask.cpu().numpy())
            tag     = f'NormTop_layer{layer_idx}_t{int(t*100)}_k{k}'
            generate_and_save_completions_for_dataset(
                cfg, model_base, [(model_base.model_block_modules[layer_idx], hook_fn)], [],
                tag, 'harmful', batch_size=batch_size, dataset=harmful_test,
                save_path="completions/norm_sweep", use_existing=use_existing
            )
            tags.append((tag, "completions/norm_sweep"))
            print(f"    L{layer_idx:02d} top t={t:.2f} k={k} ({100*k/hidden_size:.1f}% of {hidden_size})")

        for t in NORM_SWEEP_THRESHOLDS_BOT:
            mask     = get_norm_explained_mask(direction, t)
            k        = len(mask)
            bot_mask = sorted_indices[k:].copy()
            hook_fn  = make_sparse_projection_input_hook(direction, bot_mask)
            tag      = f'NormBot_layer{layer_idx}_t{int(t*100)}_k{len(bot_mask)}'
            generate_and_save_completions_for_dataset(
                cfg, model_base, [(model_base.model_block_modules[layer_idx], hook_fn)], [],
                tag, 'harmful', batch_size=batch_size, dataset=harmful_test,
                save_path="completions/norm_sweep", use_existing=use_existing
            )
            tags.append((tag, "completions/norm_sweep"))
            print(f"    L{layer_idx:02d} bot t={t:.2f} k={len(bot_mask)} ({100*len(bot_mask)/hidden_size:.1f}% of {hidden_size})")

    return tags


# ── 2. Random Orthogonal Rotation Control ─────────────────────────────────────
# Motivation: the norm-explained mask selects coordinates aligned with the
# natural basis of r*.  A null hypothesis is that *any* k coordinates work
# equally well — i.e., the mask picks up signal simply because it covers k
# dimensions of a direction that is already a good projector, not because those
# specific coordinates matter.
#
# This experiment falsifies that null by rotating r* with a random orthogonal Q
# and running the *same* top-k masking procedure on the rotated direction
# (without rotating the activations).  If the original mask's performance
# reflects genuine coordinate alignment, rotated masks at matched k should
# perform substantially worse.  If the null is correct, they should perform
# similarly.
#
# Design choices
# ──────────────
# • We match k to the norm-explained thresholds from NORM_SWEEP_THRESHOLDS_TOP
#   so results are directly comparable.
# • NUM_ROTATION_TRIALS independent Q samples per (layer, threshold) give an
#   empirical mean ± std, letting us distinguish variance from signal.
# • Seeds are fixed as (trial_index * 1000 + layer_idx) for reproducibility.
# • Activations are NOT rotated ("coordinate-only" variant). This is the most
#   conservative and interpretable test: the rotation scrambles which dimensions
#   of r* are called "top-k", while leaving the model's representation intact.

def run_rotation_control(cfg, model_base, directions,
                          sweep_layers, harmful_test,
                          batch_size, use_existing,
                          num_trials=NUM_ROTATION_TRIALS):
    """
    For each layer in sweep_layers and each norm-explained threshold, run
    num_trials rotation trials and save completions for later evaluation.

    Tags follow the pattern:
        RotCtrl_layer{L}_t{T}_trial{i}

    where T is the integer threshold percentage (e.g. 90 for t=0.9) and i is
    the trial index (0-indexed).

    Returns:
        tags_and_paths: list of (tag, save_path) pairs for evaluate_tags()
        rotation_meta:  dict mapping tag -> {'layer', 'threshold', 'trial',
                                              'k', 'k_frac', 'seed'}
                        saved to artifact_path/rotation_control/meta.json
    """
    tags_and_paths = []
    rotation_meta  = {}
    save_path      = "completions/rotation_control"

    for layer_idx in sweep_layers:
        direction   = directions[layer_idx]
        hidden_size = direction.shape[0]

        print(f"\n  Layer {layer_idx} — rotation control")

        for t in [0.95]: #NORM_SWEEP_THRESHOLDS_TOP:
            # Determine k from the *original* direction so it matches norm sweep
            orig_mask = get_norm_explained_mask(direction, t)
            k         = len(orig_mask)
            k_frac    = k / hidden_size

            for trial in range(num_trials):
                seed = trial * 1000 + layer_idx
                Q    = sample_haar_rotation(hidden_size, dtype=direction.dtype,
                                            device=direction.device, seed=seed)

                # Top-k coords of the rotated direction by magnitude
                r_rot        = Q @ direction
                rot_sorted   = r_rot.abs().argsort(descending=True)
                rot_mask_idx = rot_sorted[:k].cpu().numpy()

                hook_fn = make_rotated_sparse_projection_hook(direction, rot_mask_idx, Q)

                tag = f'RotCtrl_layer{layer_idx}_t{int(t * 100)}_trial{trial}'
                generate_and_save_completions_for_dataset(
                    cfg, model_base,
                    [(model_base.model_block_modules[layer_idx], hook_fn)], [],
                    tag, 'harmful',
                    batch_size=batch_size, dataset=harmful_test,
                    save_path=save_path, use_existing=use_existing
                )
                tags_and_paths.append((tag, save_path))
                rotation_meta[tag] = {
                    'layer':     layer_idx,
                    'threshold': t,
                    'trial':     trial,
                    'seed':      seed,
                    'k':         k,
                    'k_frac':    round(k_frac, 4),
                }
                print(f"    t={t:.2f} trial={trial} k={k} ({100*k_frac:.1f}%)")

    # Persist metadata alongside the completions
    meta_dir  = os.path.join(cfg.artifact_path(), save_path)
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, "rotation_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(rotation_meta, f, indent=2)
    print(f"\n  Rotation metadata saved to {meta_path}")

    return tags_and_paths, rotation_meta


def summarise_rotation_results(cfg, rotation_meta):
    """
    After evaluation, print a table comparing original norm-sweep ASR against
    the mean ± std of rotation-control ASR at matched (layer, threshold, k).

    Also loads the corresponding NormTop results for direct comparison and
    writes a summary JSON to artifact_path/rotation_control/summary.json.
    """
    # Collect rotation ASRs
    rot_asr = {}  # (layer, t) -> list of ASRs
    for tag, meta in rotation_meta.items():
        eval_path = os.path.join(
            cfg.artifact_path(),
            f"completions/rotation_control/harmful_{tag}_evaluations.json"
        )
        if not os.path.exists(eval_path):
            continue
        with open(eval_path) as f:
            asr = json.load(f)['llamaguard3_success_rate']
        key = (meta['layer'], meta['threshold'])
        rot_asr.setdefault(key, []).append(asr)

    # Collect original norm-sweep ASRs for comparison
    orig_asr = {}  # (layer, t) -> ASR
    for (layer_idx, t), asrs in rot_asr.items():
        # Reconstruct the NormTop tag — we need k, which is in the meta
        sample_tag = next(tag for tag, m in rotation_meta.items()
                          if m['layer'] == layer_idx and m['threshold'] == t)
        k = rotation_meta[sample_tag]['k']
        norm_tag   = f'NormTop_layer{layer_idx}_t{int(t * 100)}_k{k}'
        norm_path  = os.path.join(
            cfg.artifact_path(),
            f"completions/norm_sweep/harmful_{norm_tag}_evaluations.json"
        )
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                orig_asr[(layer_idx, t)] = json.load(f)['llamaguard3_success_rate']

    print("\n── Rotation Control Summary ─────────────────────────────────────────")
    print(f"{'Layer':>6}  {'t':>5}  {'k_frac':>7}  {'Orig ASR':>9}  "
          f"{'Rot Mean':>9}  {'Rot Std':>8}  {'Delta':>7}")
    print("─" * 70)

    summary = {}
    for (layer_idx, t) in sorted(rot_asr.keys()):
        asrs    = rot_asr[(layer_idx, t)]
        mean_r  = float(np.mean(asrs))
        std_r   = float(np.std(asrs))
        orig    = orig_asr.get((layer_idx, t), float('nan'))
        delta   = mean_r - orig if not np.isnan(orig) else float('nan')
        sample_tag = next(tag for tag, m in rotation_meta.items()
                          if m['layer'] == layer_idx and m['threshold'] == t)
        k_frac  = rotation_meta[sample_tag]['k_frac']
        print(f"{layer_idx:>6}  {t:>5.2f}  {k_frac:>7.3f}  "
              f"{orig:>9.3f}  {mean_r:>9.3f}  {std_r:>8.3f}  {delta:>+7.3f}")
        summary[f"L{layer_idx}_t{int(t*100)}"] = {
            'layer': layer_idx, 'threshold': t, 'k_frac': k_frac,
            'orig_asr': orig, 'rot_mean_asr': mean_r,
            'rot_std_asr': std_r, 'delta': delta, 'n_trials': len(asrs),
        }

    summary_path = os.path.join(cfg.artifact_path(),
                                 "completions/rotation_control/summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {summary_path}")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',           type=str,  required=True)
    parser.add_argument('--batch_size',            type=int,  default=128)
    parser.add_argument('--use_existing',          action='store_true')
    parser.add_argument('--run_norm_sweep',        action='store_true')
    parser.add_argument('--run_rotation_control',  action='store_true',
                        help='Run random orthogonal rotation control on top ASR layers')
    parser.add_argument('--num_rotation_trials',   type=int,  default=NUM_ROTATION_TRIALS,
                        help='Number of independent rotation trials per (layer, threshold)')

    args = parser.parse_args()

    cfg           = Config(model_alias=os.path.basename(args.model_path), model_path=args.model_path)
    model_name    = os.path.basename(args.model_path)
    model_base    = construct_model_base(args.model_path)
    artifact_path = cfg.artifact_path()
    os.makedirs(artifact_path, exist_ok=True)

    # Data
    harmful_train  = load_dataset_split(harmtype='mixed_harmful', split='train', instructions_only=False)
    harmless_train = load_dataset_split(harmtype='harmless',       split='train', instructions_only=True)
    harmful_test   = load_dataset_split(harmtype='mixed_harmful', split='test',  instructions_only=False)[:512]

    # Directions
    category_directions, harmless_mean = generate_and_save_category_directions(
        cfg, model_base, harmful_train, harmless_train,
        batch_size=args.batch_size, use_existing=args.use_existing
    )
    first_direction = next(iter(category_directions.values()))[1]
    num_layers      = first_direction.shape[1]
    hidden_size     = first_direction.shape[-1]
    all_layers      = list(range(1, num_layers))
    directions      = build_directions(category_directions, num_layers)

    # ── Phase 1: Sparsity — full sweep then norm sweep on top layers ─────────

    print("\nPHASE 1a: Full direction sweep (all layers)...")
    tags = run_full_sweep(
        cfg, model_base, directions, harmless_mean,
        all_layers, artifact_path, model_name,
        harmful_test, args.batch_size, args.use_existing
    )

    unload_model(model_base)
    evaluate_tags(cfg, tags, args.batch_size, args.use_existing)
    model_base = construct_model_base(args.model_path)

    print("\nPHASE 1b: Norm-explained sweep (top layers by ASR)...")
    top_layers = get_top_layers_by_asr(cfg, all_layers)
    tags = run_norm_sweep(
        cfg, model_base, directions, harmless_mean,
        top_layers, artifact_path, model_name,
        harmful_test, args.batch_size, args.use_existing
    )

    unload_model(model_base)
    evaluate_tags(cfg, tags, args.batch_size, args.use_existing)

    # ── Phase 2: Rotation control ─────────────────────────────────────────────

    if args.run_rotation_control:
        print("\nPHASE 2: Random orthogonal rotation control...")
        print(f"  Layers: {top_layers} | Trials per (layer, threshold): {args.num_rotation_trials}")
        model_base = construct_model_base(args.model_path)

        rot_tags, rotation_meta = run_rotation_control(
            cfg, model_base, directions,
            sweep_layers=top_layers,
            harmful_test=harmful_test,
            batch_size=args.batch_size,
            use_existing=args.use_existing,
            num_trials=args.num_rotation_trials,
        )

        unload_model(model_base)
        evaluate_tags(cfg, rot_tags, args.batch_size, args.use_existing)

        # Print comparison table and write summary JSON
        summarise_rotation_results(cfg, rotation_meta)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    print("\nDone.")


if __name__ == "__main__":
    main()