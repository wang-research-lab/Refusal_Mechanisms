"""Position perturbation of per-component contribution profiles (rebuttal).

Descriptive characterization of how each mechanism component's linear attribution
to the refusal direction shifts across post-instruction token positions. r* is
fixed (paper's standard layer); only the read-out position varies.

This is intentionally NOT a hypothesis test. No p-values, no significance tests
- we report Spearman rank correlation, mean |alpha|, and magnitude shift only.

Method (per model)
  1. Load r* from pipeline/runs/<alias>/generate_directions/refusal_directions.pt
     at the paper's anchor layer l*. Normalize to r_hat.
  2. Load the identified mechanism components from
     pipeline/runs/<alias>/exhaustive_set_anchor_<l*>_determinstic.json.
  3. For positions in POSITIONS_BY_MODEL[alias], register forward hooks on each
     component's module that capture the output sliced at those positions,
     accumulating sums across harmful_train and harmless_train.
  4. mean_harmful(I) - mean_harmless(I) = r_c(I) per (component, position).
  5. alpha_c(I) = dot(r_c(I), r_hat)                  # matches plot_contributions.py
                = (r_c(I) . r*) / ||r*||
  6. Report Spearman rho between alpha vectors at I=-1 vs I=-2 (and -1 vs -3 if
     used), mean |alpha| per position, % shift in mean |alpha| between positions.
  7. Reproduction check at I=-1: load the cached per-component means
     (component_mean_output_norms_deterministic_{harmful,harmless}.pt) and verify
     our recomputed alpha_c(-1) matches the cache-derived values within tolerance.

Mistral is dropped: its chat template ends with a single [/INST] token, so I=-2
falls inside the user's instruction (varies prompt-to-prompt). Phi is restricted
to {-1, -2}: its template is only <|end|><|assistant|>, so I=-3 is in user content.

Token-position audit
  Before computing anything we tokenize a sample of harmful + harmless prompts,
  read out the token IDs at each requested position, and confirm they are
  constant across prompts. Stored in JSON as position_tokens for review.

Outputs
  pipeline/runs/<alias>/position_perturbation.json  (per-model sidecar)
  pipeline/runs/position_perturbation_table.md      (combined markdown table)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from dataset.load_dataset import load_dataset_split


# Paper anchors per model (the layer at which r* is read, matching the spec).
ANCHORS: Dict[str, int] = {
    "google/gemma-2-2b-it": 13,
    "Qwen/Qwen3-4B": 19,
    "microsoft/Phi-4-mini-instruct": 14,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 17,
}

# Post-instruction positions to measure per model. Determined by chat-template depth:
#   gemma-2 ends with <end_of_turn>\n<start_of_turn>model\n   -> -1,-2,-3 all template
#   Qwen3   ends with <|im_end|>\n<|im_start|>assistant\n      -> -1,-2,-3 all template
#   Phi-4   ends with <|end|><|assistant|>                     -> only -1,-2 reliable
#   Mistral ends with [/INST]  (single special token)          -> only I=-1 reliable;
#                                                                 included for the head-sparsity
#                                                                 column only, position-shift
#                                                                 columns are reported as n/a.
POSITIONS_BY_MODEL: Dict[str, List[int]] = {
    "google/gemma-2-2b-it": [-1, -2, -3],
    "Qwen/Qwen3-4B": [-1, -2, -3],
    "microsoft/Phi-4-mini-instruct": [-1, -2],
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": [-1],
}

OUT_NAME = "position_perturbation.json"
TABLE_NAME = "position_perturbation_table.md"
RUNS_ROOT = "pipeline/runs"


# ----- artifact loaders -------------------------------------------------------

def load_r_star(cfg: Config, anchor_layer: int) -> torch.Tensor:
    p = os.path.join(cfg.artifact_path(), "generate_directions", "refusal_directions.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    if anchor_layer not in d:
        # plot_contributions.py falls back to the median key. Surface this clearly
        # so a Qwen / Phi mismatch is easy to diagnose.
        keys = sorted(d.keys())
        median = keys[len(keys) // 2]
        raise KeyError(
            f"anchor layer {anchor_layer} not in refusal_directions.pt "
            f"(keys range {keys[0]}..{keys[-1]}); plot_contributions.py default "
            f"would have used median key {median}."
        )
    return d[anchor_layer].detach().cpu().float()


def load_mechanism_components(cfg: Config, anchor_layer: int) -> List[str]:
    p = os.path.join(cfg.artifact_path(), f"exhaustive_set_anchor_{anchor_layer}_determinstic.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"mechanism JSON not found: {p}")
    return json.load(open(p))["components"]


def load_cached_component_means(cfg: Config):
    """Returns ({comp: harmful_mean}, {comp: harmless_mean}) at the paper's single
    position. Used only for the I=-1 reproduction check."""
    h = torch.load(os.path.join(cfg.artifact_path(),
                                "component_mean_output_norms_deterministic_harmful.pt"),
                   map_location="cpu", weights_only=False)
    hl = torch.load(os.path.join(cfg.artifact_path(),
                                 "component_mean_output_norms_deterministic_harmless.pt"),
                    map_location="cpu", weights_only=False)
    return h, hl


# ----- module resolution ------------------------------------------------------

def resolve_component_module(model_base, name: str):
    if name.startswith("attn_"):
        return model_base.model_attn_modules[int(name.split("_", 1)[1])]
    if name.startswith("mlp_"):
        return model_base.model_mlp_modules[int(name.split("_", 1)[1])]
    if name == "embed":
        return model_base.model.get_input_embeddings()
    raise ValueError(f"unknown component name: {name}")


# ----- token-position audit ---------------------------------------------------

def tokenize_with_chat(model_base, prompts):
    """Apply the model's instruction-format function and return CPU tensors so we
    can inspect token IDs at chosen positions."""
    fn = model_base.tokenize_instructions_fn
    out = fn(instructions=prompts)
    return out  # dict with input_ids, attention_mask (CPU at this point)


def audit_positions(model_base, prompts, positions: List[int], n: int = 8) -> Dict[str, List]:
    """Check that the token IDs at each position are constant across prompts.
    Returns {str(pos): {"id": int|None, "decoded": str|None, "constant_across_prompts": bool}}.
    """
    sample = prompts[:n]
    enc = tokenize_with_chat(model_base, sample)
    ids = enc["input_ids"]                                 # (B, S)
    tok = model_base.tokenizer
    report = {}
    for pos in positions:
        col = ids[:, pos].tolist()
        constant = all(x == col[0] for x in col)
        decoded = tok.decode([col[0]]) if constant else None
        report[str(pos)] = {
            "id": col[0] if constant else None,
            "decoded": decoded,
            "constant_across_prompts": bool(constant),
            "ids_seen": sorted(set(col)),
        }
    return report


# ----- forward-pass collection ------------------------------------------------

class _MultiPosCollector:
    """Accumulates sum of component outputs at requested positions across a forward
    pass. One instance per (component, position-set) -- light to construct, but we
    share the count across all components via a shared dict.
    """
    def __init__(self, n_positions: int, d_model: int, device: str = "cpu"):
        self.sum = torch.zeros(n_positions, d_model, dtype=torch.float32, device=device)

    def add(self, out_slice_BPD: torch.Tensor):
        # sum over batch
        self.sum += out_slice_BPD.sum(dim=0).detach().float().to(self.sum.device)


def make_hook(collector: _MultiPosCollector, positions: List[int]):
    def hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output    # (B, S, D)
        # gather requested positions -> (B, len(positions), D)
        slice_BPD = out[:, positions, :]
        collector.add(slice_BPD)
    return hook


def make_pre_hook(collector: _MultiPosCollector, positions: List[int]):
    """Forward pre-hook: captures the module's input. Used on o_proj to grab the
    multi-head concatenated tensor (B, S, n_heads*d_head) before projection, which
    is what we need for per-head decomposition."""
    def hook(module, args):
        x = args[0] if isinstance(args, tuple) else args            # (B, S, n_heads*d_head)
        slice_BPD = x[:, positions, :]
        collector.add(slice_BPD)
    return hook


def collect_component_means(model_base, prompts: List[str], components: List[str],
                            positions: List[int], batch_size: int = 8,
                            attn_block_names: List[str] | None = None,
                            max_prompts: int | None = None):
    """Run forward passes and collect:
      - mean per-component output at each requested position (per `components`)
      - if attn_block_names is given, also mean o_proj-input at each position for
        each named attn-block (used downstream for per-head decomposition)

    Returns (means, attn_in_means, count) where:
      means          : {component_name: tensor(n_positions, d_model)}
      attn_in_means  : {attn_name: tensor(n_positions, n_heads*d_head)} (empty if not requested)
    """
    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    device = next(model_base.model.parameters()).device
    d_model = model_base.model.config.hidden_size

    # post-hook collectors for component outputs
    collectors: Dict[str, _MultiPosCollector] = {
        c: _MultiPosCollector(len(positions), d_model, device="cpu") for c in components
    }
    # pre-hook collectors for o_proj inputs (per attn-block)
    attn_collectors: Dict[str, _MultiPosCollector] = {}
    if attn_block_names:
        for a in attn_block_names:
            idx = int(a.split("_", 1)[1])
            o_proj = model_base.model_attn_modules[idx].o_proj
            d_in = o_proj.weight.shape[1]                            # n_heads * d_head
            attn_collectors[a] = _MultiPosCollector(len(positions), d_in, device="cpu")

    handles = []
    for c in components:
        mod = resolve_component_module(model_base, c)
        handles.append(mod.register_forward_hook(make_hook(collectors[c], positions)))
    for a, col in attn_collectors.items():
        idx = int(a.split("_", 1)[1])
        o_proj = model_base.model_attn_modules[idx].o_proj
        handles.append(o_proj.register_forward_pre_hook(make_pre_hook(col, positions)))

    count = 0
    try:
        model_base.model.eval()
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                enc = tokenize_with_chat(model_base, batch)
                enc = {k: v.to(device) for k, v in enc.items()}
                _ = model_base.model(**enc)
                count += len(batch)
    finally:
        for h in handles:
            h.remove()

    means = {c: (collectors[c].sum / max(count, 1)) for c in components}
    attn_in_means = {a: (col.sum / max(count, 1)) for a, col in attn_collectors.items()}
    return means, attn_in_means, count


# ----- analysis ---------------------------------------------------------------

def per_position_alpha(means_harmful, means_harmless, components, r_star) -> np.ndarray:
    """Returns an array of shape (n_components, n_positions) with alpha_c(I) =
    dot(r_c(I), r_hat), where r_hat = r* / ||r*||.

    This matches plot_contributions.py's formulation (which is what Figure 4 shows).
    The spec's normalization choice differs by a constant factor ||r*||; Spearman is
    rank-invariant and magnitude-shift % is a ratio, so this choice does not affect
    the reported summary statistics.
    """
    r_hat = (r_star / (r_star.norm() + 1e-8)).float()
    rows = []
    for c in components:
        r_c = (means_harmful[c] - means_harmless[c]).float()    # (n_positions, d_model)
        a = (r_c @ r_hat).cpu().numpy()                          # (n_positions,)
        rows.append(a)
    return np.stack(rows, axis=0)                                # (n_comp, n_pos)


def get_oproj_weight(model_base, attn_name: str) -> torch.Tensor:
    """Return o_proj.weight (d_model, n_heads*d_head) on CPU/float32 for the
    named attn-block (e.g. "attn_11")."""
    idx = int(attn_name.split("_", 1)[1])
    W = model_base.model_attn_modules[idx].o_proj.weight
    return W.detach().cpu().float()


def decompose_heads_alpha(diff_PD: torch.Tensor, W_O: torch.Tensor,
                          r_hat: torch.Tensor, n_heads: int) -> np.ndarray:
    """Per-head contribution to r* per position.

    Inputs
      diff_PD : (n_positions, n_heads*d_head)         o_proj-input diff at each position
      W_O     : (d_model, n_heads*d_head)             o_proj weight matrix
      r_hat   : (d_model,)                             unit refusal direction
      n_heads : int                                    number of attention heads

    Math: o_proj(x) decomposes as sum_h W_O[:, h*Dh:(h+1)*Dh] @ x[h*Dh:(h+1)*Dh],
    so the per-head residual contribution is exact. Projecting onto r_hat gives
        alpha_{p,h} = einsum("d, d -> ", diff_p[h], W_O_columns[h] . r_hat)
    Implemented in the order (W·r̂) then (diff · result) to avoid materialising the
    full (P, H, d_model) tensor.

    Returns alpha array of shape (n_positions, n_heads).
    """
    d_head = W_O.shape[1] // n_heads
    diff_PHD = diff_PD.float().reshape(diff_PD.shape[0], n_heads, d_head)         # (P, H, Dh)
    W_MHD = W_O.float().reshape(W_O.shape[0], n_heads, d_head)                    # (M, H, Dh)
    write_dir = torch.einsum("Mhd, M -> hd", W_MHD, r_hat.float())                # (H, Dh)
    alpha_PH = torch.einsum("phd, hd -> ph", diff_PHD, write_dir)                 # (P, H)
    return alpha_PH.cpu().numpy()


def sparsity_topk(values_1d, target: float = 0.90) -> Dict:
    """Smallest k such that the top-k |values| sum to >= target * total |values|.
    Reports {k, n, k_fraction, captured_fraction}. n is the number of available units."""
    abs_vals = np.abs(np.asarray(values_1d, dtype=np.float64))
    n = int(abs_vals.size)
    total = float(abs_vals.sum())
    if n == 0 or total <= 0:
        return {"k": 0, "n": n, "k_fraction": 0.0, "captured_fraction": 0.0}
    sorted_desc = np.sort(abs_vals)[::-1]
    cum = np.cumsum(sorted_desc)
    k = int(np.searchsorted(cum, target * total) + 1)
    k = min(k, n)
    return {
        "k": k,
        "n": n,
        "k_fraction": k / n,
        "captured_fraction": float(cum[k - 1] / total),
    }


def reproduction_check_pos_minus_one(cfg: Config, components: List[str], r_star: torch.Tensor,
                                     our_alpha_minus1: np.ndarray) -> Dict:
    """Project the CACHED per-component diff (paper's stored means) onto r_hat and
    compare to our recomputed alpha at I=-1. The cached means are at the paper's
    single position, which is I=-1."""
    try:
        h, hl = load_cached_component_means(cfg)
    except FileNotFoundError as e:
        return {"available": False, "reason": str(e)}
    r_hat = (r_star / (r_star.norm() + 1e-8)).float()
    cached_alpha = []
    missing = []
    for c in components:
        if c not in h or c not in hl:
            missing.append(c); cached_alpha.append(np.nan); continue
        diff = (h[c].float() - hl[c].float())
        cached_alpha.append(torch.dot(diff, r_hat).item())
    cached_alpha = np.array(cached_alpha, dtype=np.float64)
    valid = ~np.isnan(cached_alpha)
    if not valid.any():
        return {"available": True, "matched_components": 0, "missing": missing,
                "max_abs_diff": None, "passed": False}
    max_abs_diff = float(np.max(np.abs(cached_alpha[valid] - our_alpha_minus1[valid])))
    rel = max_abs_diff / max(float(np.max(np.abs(cached_alpha[valid]))), 1e-12)
    return {
        "available": True,
        "matched_components": int(valid.sum()),
        "missing": missing,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": rel,
        "passed": bool(rel < 1e-2),    # < 1% relative
    }


def build_record(alias: str, anchor_layer: int, components: List[str],
                 positions: List[int], alphas: np.ndarray, pos_audit, repro,
                 n_harmful: int, n_harmless: int,
                 attn_heads: Dict | None = None,
                 pooled_head_sparsity: Dict | None = None) -> Dict:
    """Assemble the per-model JSON record. alphas is (n_comp, n_pos).
    attn_heads: optional dict keyed by attn-block name with per-head decomposition.
    pooled_head_sparsity: optional dict keyed by position str with pooled sparsity stats.
    """
    pos_idx = {p: k for k, p in enumerate(positions)}
    rec = {
        "model_name": alias,
        "r_star_layer": anchor_layer,
        "positions_used": positions,
        "position_tokens": pos_audit,
        "n_harmful_prompts": n_harmful,
        "n_harmless_prompts": n_harmless,
        "components": {},
    }
    for ci, c in enumerate(components):
        rec["components"][c] = {f"alpha_{p}": float(alphas[ci, pos_idx[p]]) for p in positions}
    if attn_heads is not None:
        rec["attn_heads"] = attn_heads
    if pooled_head_sparsity is not None:
        rec["pooled_head_sparsity"] = pooled_head_sparsity

    abs_means = {p: float(np.mean(np.abs(alphas[:, pos_idx[p]]))) for p in positions}
    for p in positions:
        rec[f"mean_abs_magnitude_{p}"] = abs_means[p]

    base = -1
    for p in positions:
        if p == base: continue
        rec[f"spearman_{base}_vs_{p}"] = float(
            spearmanr(alphas[:, pos_idx[base]], alphas[:, pos_idx[p]]).statistic
        )
        # signed % shift (positive = larger magnitude at p, negative = smaller)
        if abs_means[base] > 0:
            rec[f"magnitude_shift_pct_{base}_to_{p}"] = float(
                100.0 * (abs_means[p] - abs_means[base]) / abs_means[base]
            )
        else:
            rec[f"magnitude_shift_pct_{base}_to_{p}"] = None

    rec["reproduction_check"] = repro
    return rec


# ----- markdown table ---------------------------------------------------------

def emit_table(records: List[Dict], out_path: str):
    has_minus3 = any(-3 in r["positions_used"] for r in records)
    head1 = ["Model", "ρ (-1 vs -2)", "Mean |α| at -1", "Mean |α| at -2", "Magnitude shift -1→-2"]
    if has_minus3:
        head1 += ["ρ (-1 vs -3)", "Mean |α| at -3", "Magnitude shift -1→-3"]
    head1 += ["Head sparsity @ -1"]
    lines = ["| " + " | ".join(head1) + " |",
             "|" + "|".join(["-"] * len(head1)) + "|"]

    def fmt_alpha(x): return "—" if x is None else f"{x:.3f}"
    def fmt_pct(x):   return "—" if x is None else f"{x:+.1f}%"
    def fmt_rho(x):   return "—" if x is None else f"{x:.2f}"
    def fmt_sparsity(s):
        if s is None or s.get("n", 0) == 0: return "—"
        return f"top {s['k']}/{s['n']} ({s['k_fraction']:.0%}) → {s['captured_fraction']:.0%}"

    for r in records:
        row = [r["model_name"],
               fmt_rho(r.get("spearman_-1_vs_-2")),
               fmt_alpha(r.get("mean_abs_magnitude_-1")),
               fmt_alpha(r.get("mean_abs_magnitude_-2")),
               fmt_pct(r.get("magnitude_shift_pct_-1_to_-2"))]
        if has_minus3:
            if -3 in r["positions_used"]:
                row += [fmt_rho(r.get("spearman_-1_vs_-3")),
                        fmt_alpha(r.get("mean_abs_magnitude_-3")),
                        fmt_pct(r.get("magnitude_shift_pct_-1_to_-3"))]
            else:
                row += ["n/a", "n/a", "n/a"]
        row += [fmt_sparsity((r.get("pooled_head_sparsity") or {}).get("-1"))]
        lines.append("| " + " | ".join(row) + " |")

    open(out_path, "w").write("\n".join(lines) + "\n")


# ----- main -------------------------------------------------------------------

def run_one(model_path: str, batch_size: int, max_prompts: int | None) -> Dict:
    if model_path not in ANCHORS:
        raise SystemExit(f"unsupported model_path: {model_path}. Supported: {list(ANCHORS)}")
    anchor = ANCHORS[model_path]
    positions = POSITIONS_BY_MODEL[model_path]
    alias = os.path.basename(model_path)

    print(f"[{alias}] anchor_layer={anchor} positions={positions}")
    cfg = Config(model_alias=alias, model_path=model_path)
    model_base = construct_model_base(model_path)

    # Data: same sets the paper uses for Figure 4 component attributions.
    harmful  = load_dataset_split("mixed_harmful", "train", instructions_only=False)
    harmless = load_dataset_split("harmless",       "train", instructions_only=True)[:len(harmful)]
    if max_prompts is not None:
        harmful  = harmful[:max_prompts]
        harmless = harmless[:max_prompts]
    print(f"[{alias}] harmful={len(harmful)} harmless={len(harmless)}")

    # r* at the paper's anchor layer.
    r_star = load_r_star(cfg, anchor)
    print(f"[{alias}] r_star norm = {r_star.norm().item():.4f}")

    components = load_mechanism_components(cfg, anchor)
    print(f"[{alias}] mechanism components ({len(components)}): {components}")

    # Token-position audit.
    audit = audit_positions(model_base, [d['instruction'] if isinstance(d, dict) else d
                                         for d in harmful], positions, n=12)
    print(f"[{alias}] position audit:")
    for k, v in audit.items():
        print(f"    I={k}: id={v['id']}  decoded={v['decoded']!r}  "
              f"constant_across_prompts={v['constant_across_prompts']}")

    # Mechanism attn-blocks to decompose to heads. mlp/embed have no clean per-unit
    # interpretation under superposition; we restrict head decomposition to attn.
    attn_names = [c for c in components if c.startswith("attn_")]
    n_heads = int(model_base.model.config.num_attention_heads)
    o_proj_weights = {a: get_oproj_weight(model_base, a) for a in attn_names}
    if attn_names:
        d_head = o_proj_weights[attn_names[0]].shape[1] // n_heads
        print(f"[{alias}] decomposing {len(attn_names)} attn-blocks into "
              f"n_heads={n_heads} d_head={d_head}")

    # Forward passes (component outputs + o_proj inputs in one sweep per side).
    print(f"[{alias}] collecting component means at positions {positions} ...")
    means_h,  attn_in_h,  n_h  = collect_component_means(
        model_base, harmful,  components, positions, batch_size, attn_block_names=attn_names)
    means_hl, attn_in_hl, n_hl = collect_component_means(
        model_base, harmless, components, positions, batch_size, attn_block_names=attn_names)

    alphas = per_position_alpha(means_h, means_hl, components, r_star)

    pos_idx = {p: k for k, p in enumerate(positions)}
    repro = reproduction_check_pos_minus_one(cfg, components, r_star, alphas[:, pos_idx[-1]])
    print(f"[{alias}] reproduction at I=-1: {repro}")

    # Per-head decomposition for each mechanism attn-block + sparsity stats.
    r_hat = r_star / (r_star.norm() + 1e-8)
    attn_heads = {}
    for a in attn_names:
        diff_PD = (attn_in_h[a] - attn_in_hl[a])                      # (P, n_heads*d_head)
        alpha_PH = decompose_heads_alpha(diff_PD, o_proj_weights[a], r_hat, n_heads)
        attn_heads[a] = {
            "n_heads": int(n_heads),
            "d_head": int(o_proj_weights[a].shape[1] // n_heads),
            "alpha":     {str(p): alpha_PH[pos_idx[p], :].tolist()   for p in positions},
            "sparsity":  {str(p): sparsity_topk(alpha_PH[pos_idx[p], :], target=0.90)
                                                                         for p in positions},
        }

    # Pooled sparsity across ALL mechanism attn-block heads (analog to component sparsity).
    pooled_head_sparsity = {}
    for p in positions:
        pooled = np.concatenate(
            [np.array(attn_heads[a]["alpha"][str(p)]) for a in attn_names]
        ) if attn_names else np.array([])
        pooled_head_sparsity[str(p)] = sparsity_topk(pooled, target=0.90)
    if attn_names:
        s = pooled_head_sparsity[str(-1)]
        print(f"[{alias}] pooled head sparsity @ I=-1: top-{s['k']}/{s['n']} heads "
              f"({s['k_fraction']:.0%}) carry {s['captured_fraction']:.0%} of |α|")

    rec = build_record(alias, anchor, components, positions, alphas, audit, repro, n_h, n_hl,
                       attn_heads=attn_heads, pooled_head_sparsity=pooled_head_sparsity)

    out_path = os.path.join(cfg.artifact_path(), OUT_NAME)
    json.dump(rec, open(out_path, "w"), indent=2)
    print(f"[{alias}] wrote {out_path}")

    # cleanup
    del model_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", default=None,
                   help="If given, run just this model. Else run all in ANCHORS.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Cap on the number of harmful / harmless prompts (smoke test).")
    p.add_argument("--report", action="store_true",
                   help="Aggregate existing per-model JSONs into the combined markdown table; "
                        "skips running forward passes.")
    p.add_argument("--out", default=os.path.join(RUNS_ROOT, TABLE_NAME),
                   help="Combined markdown table path.")
    args = p.parse_args()

    if args.report:
        records = []
        for mp in ANCHORS:
            alias = os.path.basename(mp)
            f = os.path.join(RUNS_ROOT, alias, OUT_NAME)
            if os.path.exists(f):
                records.append(json.load(open(f)))
        emit_table(records, args.out)
        print(f"aggregated {len(records)} model record(s) -> {args.out}")
        return

    targets = [args.model_path] if args.model_path else list(ANCHORS.keys())
    records = []
    for mp in targets:
        records.append(run_one(mp, args.batch_size, args.max_prompts))

    emit_table(records, args.out)
    print(f"wrote combined markdown table -> {args.out}")


if __name__ == "__main__":
    main()
