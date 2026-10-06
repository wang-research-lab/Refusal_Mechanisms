"""
save_ortho_weights.py
=====================
Bake a refusal-direction ablation directly into a model's weights and write the
result out as a standard HF checkpoint, so it can be served with vLLM (which
cannot run the Python activation hooks used by sweep_layers.py /
steer_components.py).

Two settings, mirroring the paper's two interventions:

  full       — "full refusal steering". Project the (dense) anchor-layer
               refusal direction r* out of EVERY residual-writing weight in
               layers [0, anchor): each layer's self_attn.o_proj and
               mlp.down_proj (and optionally the token embedding). This is the
               weight-baked equivalent of the full-direction projection
               ablation at the input of `anchor` (Arditi et al.), restricted to
               the layers that feed the anchor.

  mechanism  — "identified mechanism", sparse. Project r* out of ONLY the
               circuit components saved by steer_components.py in
               runs/<alias>/exhaustive_set_anchor_<L>_determinstic.json
               (e.g. mlp_6, attn_12, ...), and only along the top-k coordinates
               returned by get_norm_explained_mask(r*, sparse_threshold).

Both are exact weight-space equivalents of the hooks in steer_components.py.
For a residual-writing linear W (output added to the residual stream) and unit
direction d, the output hook computes  h_out = h - (mask ⊙ d)(d·h).  Pushing
that through W gives the effective weight

    W' = W - (mask ⊙ d)(dᵀ W)

which is exactly get_(masked_)orthogonalized_matrix(W.T, d, mask).T. With
mask = all-ones this reduces to the repo's existing get_orthogonalized_matrix.

Usage
-----
    python -m pipeline.save_ortho_weights \
        --model_path google/gemma-2-2b-it --setting full
    python -m pipeline.save_ortho_weights \
        --model_path google/gemma-2-2b-it --setting mechanism --sparse_threshold 0.95

Output goes to  $HF_HUB_CACHE/<alias>-ortho-<tag>  (default
$HF_HUB_CACHE/...), which serve_and_eval.sh auto-translates to the
in-container /models/<...> path.
"""

import argparse
import glob
import json
import os
import re

import torch

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.utils import get_orthogonalized_matrix


# ── Orthogonalization ──────────────────────────────────────────────────────────

def get_masked_orthogonalized_matrix(matrix, vec, mask=None):
    """
    Project `vec` out of `matrix` along the last dim, but only on the coordinates
    selected by `mask` (a bool tensor over d_model). mask=None ⇒ dense projection
    (identical to utils.get_orthogonalized_matrix).
    """
    if mask is None:
        return get_orthogonalized_matrix(matrix, vec)
    vec = vec / (vec.norm() + 1e-8)
    vec = vec.to(matrix)
    proj = (matrix @ vec).unsqueeze(-1) * vec          # (..., d_model)
    proj = proj * mask.to(matrix.device, matrix.dtype)  # zero out non-mask coords
    return matrix - proj


def orthogonalize_linear_(module, direction, mask, attr):
    """In-place: bake the (masked) projection into module.<attr>.weight."""
    W = getattr(module, attr).weight
    W.data = get_masked_orthogonalized_matrix(W.data.T, direction, mask).T


def orthogonalize_embedding_(embed_module, direction, mask):
    """In-place: project the direction out of token embeddings (rows are d_model)."""
    embed_module.weight.data = get_masked_orthogonalized_matrix(
        embed_module.weight.data, direction, mask
    )


# ── Direction loading ──────────────────────────────────────────────────────────

def load_directions(run_dir):
    """
    Reconstruct the per-layer refusal directions exactly as
    sweep_layers.build_directions / steer_components.build_directions do:
    average n * cat_dir[-1, layer] over categories.

    Returns dict {layer_idx: tensor[d_model]}.
    """
    cat_means_path = os.path.join(run_dir, "generate_directions", "cat_means.pt")
    if not os.path.exists(cat_means_path):
        raise FileNotFoundError(
            f"No cat_means.pt under {run_dir}/generate_directions/. "
            "Run sweep_layers.py / steer_components.py first."
        )
    cat_means = torch.load(cat_means_path, weights_only=False)
    first_dir = next(iter(cat_means.values()))[1]
    num_layers = first_dir.shape[1]
    return {
        layer: torch.mean(
            torch.stack([n * direction[-1, layer] for _, (n, direction) in cat_means.items()],
                        dim=0),
            dim=0,
        )
        for layer in range(1, num_layers)
    }


def norm_explained_mask(direction, energy_threshold):
    """Bool mask over d_model selecting the fewest coords explaining
    `energy_threshold` of the squared L2 norm (same as steer_components)."""
    sorted_idx = direction.abs().argsort(descending=True)
    squared = direction[sorted_idx] ** 2
    cumulative = squared.cumsum(0) / squared.sum()
    k = int((cumulative < energy_threshold).sum().item()) + 1
    keep = sorted_idx[:k]
    mask = torch.zeros(direction.shape[0], dtype=torch.bool, device=direction.device)
    mask[keep] = True
    return mask, k


# ── Circuit / anchor discovery ──────────────────────────────────────────────────

def find_circuit_json(run_dir, anchor_layer=None):
    """Locate the saved deterministic circuit JSON for this model."""
    pattern = os.path.join(run_dir, "exhaustive_set_anchor_*_determinstic.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No deterministic circuit json (exhaustive_set_anchor_*_determinstic.json) "
            f"under {run_dir}. Run steer_components.py first."
        )
    if anchor_layer is not None:
        want = os.path.join(run_dir, f"exhaustive_set_anchor_{anchor_layer}_determinstic.json")
        if want in matches:
            return want
        raise FileNotFoundError(f"No circuit json for anchor {anchor_layer} (found {matches}).")
    if len(matches) > 1:
        raise ValueError(f"Multiple circuit jsons in {run_dir}; pass --anchor_layer to disambiguate: {matches}")
    return matches[0]


def anchor_from_path(path):
    m = re.search(r"exhaustive_set_anchor_(\d+)_determinstic\.json$", path)
    return int(m.group(1)) if m else None


# ── Model module accessors ───────────────────────────────────────────────────────

def get_embed_module(model_base):
    # get_input_embeddings() is implemented by every HF CausalLM / multimodal
    # wrapper (incl. Mistral3ForConditionalGeneration), so it sidesteps the
    # per-family module-path differences.
    return model_base.model.get_input_embeddings()


def resolve_components(model_base, setting, anchor_layer, run_dir):
    """
    Return a list of (name, kind, module_or_None) to orthogonalize, where kind is
    'attn' | 'mlp' | 'embed'. For 'full' this is every attn/mlp in [0, anchor)
    (+ embed if requested); for 'mechanism' it is exactly the circuit components.
    """
    if setting == "full":
        names = []
        for i in range(anchor_layer):
            names.append(f"attn_{i}")
            names.append(f"mlp_{i}")
    else:  # mechanism
        circuit_path = find_circuit_json(run_dir, anchor_layer)
        with open(circuit_path) as f:
            names = json.load(f)["components"]
        print(f"  Loaded {len(names)} circuit components from {circuit_path}")

    comps = []
    for name in names:
        if name == "embed":
            comps.append((name, "embed", None))
        elif name.startswith("attn_"):
            idx = int(name.split("_")[1])
            comps.append((name, "attn", model_base.model_attn_modules[idx]))
        elif name.startswith("mlp_"):
            idx = int(name.split("_")[1])
            comps.append((name, "mlp", model_base.model_mlp_modules[idx]))
        else:
            print(f"  [skip] unrecognized component name: {name}")
    return comps


# ── Tied-embedding handling ─────────────────────────────────────────────────────

def untie_lm_head_if_needed(model_base):
    """
    If the input embedding and lm_head share storage, give lm_head its own copy
    of the ORIGINAL (un-orthogonalized) weights before we touch the embedding, so
    that orthogonalizing embed_tokens does not corrupt the unembedding (the hooks
    only ablate the embedding-output path, never lm_head).
    """
    model = model_base.model
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is None or inp is None:
        return False
    if out.weight is not inp.weight:
        return False  # already untied
    print("  [embed] weights tied to lm_head — untying & preserving original lm_head")
    out.weight = torch.nn.Parameter(inp.weight.data.clone())
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False
        if hasattr(model.config, "get_text_config"):
            try:
                model.config.get_text_config().tie_word_embeddings = False
            except Exception:
                pass
    return True


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True,
                   help="HF id or local path of the BASE model (e.g. google/gemma-2-2b-it).")
    p.add_argument("--setting", required=True, choices=["full", "mechanism"])
    p.add_argument("--anchor_layer", type=int, default=None,
                   help="Anchor layer. Defaults to the layer in the deterministic circuit json.")
    p.add_argument("--sparse_threshold", type=float, default=0.95,
                   help="Norm-explained energy threshold for the mechanism's coordinate mask.")
    p.add_argument("--include_embed", action="store_true",
                   help="(full only) Also orthogonalize the token embedding. Off by default so "
                        "full vs mechanism is compared over attn/mlp residual writers and tied "
                        "lm_heads are left intact.")
    p.add_argument("--source_alias", default=None,
                   help="runs/ alias to read directions & circuit from (default basename(model_path)).")
    p.add_argument("--out_root", default=os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")),
                   help="Directory to write the orthogonalized checkpoint into.")
    p.add_argument("--out_dir", default=None, help="Full output path (overrides --out_root + auto name).")
    p.add_argument("--dry_run", action="store_true", help="Build & verify but do not save_pretrained.")
    args = p.parse_args()

    alias = args.source_alias or os.path.basename(args.model_path.rstrip("/"))
    cfg = Config(model_alias=alias, model_path=args.model_path)
    run_dir = cfg.artifact_path()
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"No runs dir for alias '{alias}' at {run_dir}.")

    # Resolve anchor layer (from circuit json if not given).
    anchor_layer = args.anchor_layer
    if anchor_layer is None:
        anchor_layer = anchor_from_path(find_circuit_json(run_dir))
    print(f"Model: {args.model_path}\nAlias: {alias}\nSetting: {args.setting}\nAnchor layer: {anchor_layer}")

    # Output dir + name.
    if args.out_dir:
        out_dir = args.out_dir
    else:
        if args.setting == "full":
            tag = f"ortho-full-L{anchor_layer}"
        else:
            tag = f"ortho-mech-e{int(round(args.sparse_threshold * 100))}-L{anchor_layer}"
        out_dir = os.path.join(args.out_root, f"{alias}-{tag}")
    print(f"Output dir: {out_dir}")

    # Direction + mask.
    directions = load_directions(run_dir)
    if anchor_layer not in directions:
        raise KeyError(f"Anchor layer {anchor_layer} not in directions (have {sorted(directions)[:3]}…).")
    direction = directions[anchor_layer]
    if args.setting == "mechanism":
        mask, k = norm_explained_mask(direction, args.sparse_threshold)
        print(f"Sparse mask: keep {k}/{direction.shape[0]} coords "
              f"({100 * k / direction.shape[0]:.1f}%) at energy={args.sparse_threshold}")
    else:
        mask = None
        print("Dense projection (full direction, all coordinates).")

    # Load model (native dtype, device_map=auto, grads off) via the family factory.
    print("\nLoading model…")
    model_base = construct_model_base(args.model_path)
    model_base.model.requires_grad_(False)
    direction = direction.to(torch.float32)

    comps = resolve_components(model_base, args.setting, anchor_layer, run_dir)
    if args.setting == "full" and args.include_embed:
        comps = [("embed", "embed", None)] + comps

    # Bake the projection in.
    print(f"\nOrthogonalizing {len(comps)} components…")
    d_unit = (direction / (direction.norm() + 1e-8))
    max_resid = 0.0
    untied = False
    for name, kind, module in comps:
        if kind == "embed":
            untied = untie_lm_head_if_needed(model_base) or untied
            orthogonalize_embedding_(get_embed_module(model_base), direction, mask)
            continue
        attr = "o_proj" if kind == "attn" else "down_proj"
        if not hasattr(module, attr):
            raise AttributeError(f"{name}: module has no '{attr}' (got {type(module).__name__}).")
        orthogonalize_linear_(module, direction, mask, attr)
        # Sanity: residual projection of the modified write-weight onto d̂.
        W = getattr(module, attr).weight.data
        resid = (d_unit.to(W.device, torch.float32) @ W.float()).abs().max().item()
        max_resid = max(max_resid, resid)

    if mask is None:
        print(f"  Verify (dense): max |d̂ᵀW'| over modified attn/mlp = {max_resid:.3e} (≈0 expected)")
    else:
        kept = (d_unit ** 2)[mask.cpu()].sum().item()
        print(f"  Verify (sparse): mask captures {kept:.4f} of ‖d̂‖²; "
              f"residual d̂ᵀW' scaled by ≈{1 - kept:.4f}, max={max_resid:.3e}")

    if args.dry_run:
        print("\n[dry_run] skipping save.")
        return

    # Some family loaders mutate model.config to a sub-config (e.g. MistralModel
    # sets model.config = config.text_config). Restore the real top-level config
    # so the written config.json matches the actual model class / state dict.
    from transformers import AutoConfig
    try:
        orig_cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        if type(orig_cfg).__name__ != type(model_base.model.config).__name__:
            print(f"  Restoring full config {type(orig_cfg).__name__} "
                  f"(loader had swapped in {type(model_base.model.config).__name__}).")
            model_base.model.config = orig_cfg
    except Exception as e:
        print(f"  [warn] could not restore config ({e}); saving with current config.")

    # If we untied lm_head (to keep it un-orthogonalized while ablating embed), the
    # restored orig_cfg re-introduces tie_word_embeddings=True. vLLM trusts that and
    # ties lm_head to the orthogonalized embedding -> corrupted logits -> gibberish.
    # Re-assert the untied state so the written config matches the saved weights.
    if untied:
        model_base.model.config.tie_word_embeddings = False
        if hasattr(model_base.model.config, "get_text_config"):
            try:
                model_base.model.config.get_text_config().tie_word_embeddings = False
            except Exception:
                pass
        print("  [embed] set tie_word_embeddings=False (lm_head untied from ablated embed).")

    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving checkpoint → {out_dir}")
    model_base.model.save_pretrained(out_dir, safe_serialization=True)
    model_base.tokenizer.save_pretrained(out_dir)

    # Multimodal models (e.g. Mistral3) need an image-processor config in the dir
    # or vLLM can't build the processor and fails on load ("Can't load image
    # processor ... preprocessor_config.json"). save_pretrained on the LM doesn't
    # write one, and some repos (Mistral-Small-3.2) ship none at all — so try the
    # model's own repo first, then a loader-declared fallback (MistralModel points
    # at 3.1, same source it borrows the tokenizer from). For text-only evals the
    # image processor is never invoked; it just has to load.
    if getattr(model_base.model.config, "vision_config", None) is not None:
        import tempfile, shutil
        from transformers import AutoImageProcessor, AutoProcessor
        sources = [args.model_path, getattr(model_base, "processor_source", None)]
        last_err = None
        for src in [s for s in sources if s]:
            try:
                tmp = tempfile.mkdtemp()
                AutoImageProcessor.from_pretrained(src).save_pretrained(tmp)  # preprocessor_config.json
                try:
                    AutoProcessor.from_pretrained(src).save_pretrained(tmp)   # processor_config.json
                except Exception:
                    pass
                # copy the image/processor configs + chat template; leave the tokenizer
                # files alone. (Borrowed tokenizers like Mistral's 3.1 keep their chat
                # template in chat_template.jinja, not tokenizer_config.json — without it
                # vLLM errors: "default chat template is no longer allowed".)
                for f in ("preprocessor_config.json", "processor_config.json", "chat_template.jinja"):
                    s = os.path.join(tmp, f)
                    if os.path.exists(s):
                        shutil.copy2(s, os.path.join(out_dir, f))
                print(f"  Wrote image processor from {src}")
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            print(f"  [warn] could not write an image processor ({last_err}); "
                  f"multimodal serve will need preprocessor_config.json added manually.")
    print("Done.")
    print(f"\nServe with:\n  ./serve_and_eval.sh {out_dir} "
          f"{os.path.basename(out_dir)} <parser> <port> <gpus>")


if __name__ == "__main__":
    main()
