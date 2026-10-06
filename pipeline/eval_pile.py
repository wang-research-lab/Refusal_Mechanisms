"""
eval_pile.py
============
Quick capability/fluency check: CE loss + perplexity on the Pile for three
steering conditions of one model, on a single loaded model (no reload):

  base        — no intervention.
  refusal     — "full refusal steering": dense projection of the anchor-layer
                refusal direction out of every attn.o_proj / mlp.down_proj
                output in layers [0, anchor).
  mechanism   — the identified circuit (from steer_components' deterministic
                set), sparse: project only along the norm-explained top-k coords.

These runtime output hooks are numerically identical to the weights baked by
save_ortho_weights.py (same W' = W - (mask⊙d̂)(d̂ᵀW) applied to the same
modules), so the numbers correspond to the served orthogonalized checkpoints
without needing them on disk.

Usage:
    python -m pipeline.eval_pile --model_path google/gemma-2-2b-it
    python -m pipeline.eval_pile --model_path Qwen/Qwen3-4B --n_batches 64
"""

import argparse
import json
import os

import torch

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_linear_direction_ablation_output_hook
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.save_ortho_weights import (
    load_directions, norm_explained_mask, find_circuit_json, anchor_from_path,
    get_embed_module,
)


def make_sparse_projection_output_hook(direction, mask):
    """Forward hook: h_out = h - (mask ⊙ d̂)(d̂·h). Identical to the mechanism
    hook in steer_components.py."""
    def hook_fn(module, inp, output):
        act = output[0] if isinstance(output, tuple) else output
        d = direction.to(device=act.device, dtype=act.dtype)
        d = d / (d.norm() + 1e-8)
        proj = (act @ d).unsqueeze(-1) * d
        proj = proj * mask.to(act.device, act.dtype)
        act_mod = act - proj
        return (act_mod, *output[1:]) if isinstance(output, tuple) else act_mod
    return hook_fn


def build_hook_sets(model_base, setting, direction, mask, anchor_layer, run_dir):
    """Return a list of (module, forward_hook) for the given setting."""
    if setting == "base":
        return []

    if setting == "refusal":           # full, dense, up to anchor
        abl = get_linear_direction_ablation_output_hook(direction)
        hooks = []
        for i in range(anchor_layer):
            hooks.append((model_base.model_attn_modules[i], abl))
            hooks.append((model_base.model_mlp_modules[i], abl))
        return hooks

    # mechanism: sparse, circuit components only
    with open(find_circuit_json(run_dir, anchor_layer)) as f:
        names = json.load(f)["components"]
    sparse = make_sparse_projection_output_hook(direction, mask)
    hooks = []
    for name in names:
        if name == "embed":
            hooks.append((get_embed_module(model_base), sparse))
        elif name.startswith("attn_"):
            hooks.append((model_base.model_attn_modules[int(name.split("_")[1])], sparse))
        elif name.startswith("mlp_"):
            hooks.append((model_base.model_mlp_modules[int(name.split("_")[1])], sparse))
    return hooks


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--anchor_layer", type=int, default=None)
    p.add_argument("--sparse_threshold", type=float, default=0.95)
    p.add_argument("--variants", default="base,refusal,mechanism",
                   help="Comma-separated subset of base,refusal,mechanism.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_batches", type=int, default=64,
                   help="Pile batches per variant (-1 = stream the whole split).")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--source_alias", default=None)
    p.add_argument("--out", default=None, help="Output JSON (default runs/<alias>/pile_loss.json).")
    args = p.parse_args()

    alias = args.source_alias or os.path.basename(args.model_path.rstrip("/"))
    cfg = Config(model_alias=alias, model_path=args.model_path)
    run_dir = cfg.artifact_path()
    anchor = args.anchor_layer or anchor_from_path(find_circuit_json(run_dir))
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    directions = load_directions(run_dir)
    direction = directions[anchor].to(torch.float32)
    mask, k = norm_explained_mask(direction, args.sparse_threshold)
    print(f"Model: {args.model_path} | anchor {anchor} | sparse k={k}/{direction.shape[0]} "
          f"@ e={args.sparse_threshold}")

    model_base = construct_model_base(args.model_path)
    model_base.model.requires_grad_(False)

    results = {}
    for setting in variants:
        hooks = build_hook_sets(model_base, setting, direction, mask, anchor, run_dir)
        print(f"\n── {setting}  ({len(hooks)} hooks) ──")
        with torch.inference_mode():
            res = evaluate_loss(
                model_base, fwd_hooks=hooks, dataset_labels=["pile"],
                batch_size=args.batch_size, n_batches=args.n_batches,
                max_seq_length=args.max_seq_length,
            )
        results[setting] = res["pile"]

    out = args.out or os.path.join(run_dir, "pile_loss.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    summary = {"model": args.model_path, "anchor_layer": anchor,
               "sparse_threshold": args.sparse_threshold, "sparse_k": k,
               "n_batches": args.n_batches, "batch_size": args.batch_size,
               "max_seq_length": args.max_seq_length, "results": results}
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 56)
    print(f"PILE CE LOSS / PERPLEXITY — {alias}")
    print("=" * 56)
    print(f"{'variant':>12}  {'ce_loss':>9}  {'perplexity':>11}")
    for v in variants:
        r = results[v]
        print(f"{v:>12}  {r['ce_loss']:>9.4f}  {r['perplexity']:>11.3f}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
