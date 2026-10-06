"""
Does representation steering survive quantization?

Loads the refusal directions cached under runs/<model>/generate_directions/ by
sweep_layers.py, then reloads a single model at multiple quantization presets
and runs the *same* full-direction projection-ablation hook at the model's
top-ASR layer. Each (precision) variant produces a baseline-uncontrolled set
of completions and a steered set; both are scored with LlamaGuard-3 and the
deltas are written to a summary JSON.

Supported precisions (override via --precisions, comma-separated):
    bf16   — torch.bfloat16, no quantization (reference)
    fp16   — torch.float16, no quantization
    int8   — BitsAndBytesConfig(load_in_8bit=True), LLM.int8()
    nf4    — BitsAndBytesConfig(load_in_4bit=True, quant_type='nf4')
    fp4    — BitsAndBytesConfig(load_in_4bit=True, quant_type='fp4')
    fp8    — FineGrainedFP8Config() (native E4M3, weight-only)
    ao_int4 — TorchAoConfig('int4_weight_only')
    ao_int8 — TorchAoConfig('int8_weight_only')
    ao_fp8  — TorchAoConfig('float8_weight_only')

Example:
    python -m pipeline.quant_steering_sweep \
        --model_path google/gemma-2-2b-it \
        --precisions bf16,int8,nf4,fp4,fp8 \
        --num_test 128 --batch_size 16

Output goes to runs/{model_alias}{alias_suffix}/, so the default
'-quant' suffix keeps quantization results separate from the main
sweep_layers.py artifacts. Refusal directions are *read* from the
plain runs/{model_alias}/ tree (via --source_alias, defaulting to
basename(model_path)), so we never re-generate them.
"""

import argparse
import gc
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.completions_helper import (
    evaluate_completions_and_save_results_for_dataset,
    generate_and_save_completions_for_dataset,
)
from pipeline.submodules.evaluate_jailbreak import unload_llamaguard_model
from pipeline.sweep_layers import (
    build_directions,
    generate_and_save_category_directions,
    make_sparse_projection_input_hook,
    unload_model,
)
from dataset.load_dataset import load_dataset_split


SUPPORTED_PRECISIONS = (
    "bf16", "fp16", "int8", "nf4", "fp4", "fp8", "ao_int4", "ao_int8", "ao_fp8",
)


# ── Quantization config factory ────────────────────────────────────────────────

def build_quant_kwargs(precision: str) -> Dict:
    """Return kwargs to pass to AutoModelForCausalLM.from_pretrained for a precision tag."""
    p = precision.lower()
    common = dict(device_map="auto", attn_implementation="eager")

    if p == "bf16":
        return {**common, "torch_dtype": torch.bfloat16}
    if p == "fp16":
        return {**common, "torch_dtype": torch.float16}

    if p in ("int8", "nf4", "fp4"):
        from transformers import BitsAndBytesConfig
        if p == "int8":
            qc = BitsAndBytesConfig(load_in_8bit=True)
        else:
            qc = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=p,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        return {**common, "quantization_config": qc, "torch_dtype": torch.bfloat16}

    if p == "fp8":
        from transformers import FineGrainedFP8Config
        return {**common, "quantization_config": FineGrainedFP8Config(),
                "torch_dtype": torch.bfloat16}

    if p in ("ao_int4", "ao_int8", "ao_fp8"):
        from transformers import TorchAoConfig
        mapping = {
            "ao_int4": "int4_weight_only",
            "ao_int8": "int8_weight_only",
            "ao_fp8":  "float8_weight_only",
        }
        return {**common, "quantization_config": TorchAoConfig(mapping[p]),
                "torch_dtype": torch.bfloat16}

    raise ValueError(f"Unknown precision: {precision!r} (supported: {SUPPORTED_PRECISIONS})")


# ── Model construction at arbitrary precision ─────────────────────────────────

def construct_model_base_with_precision(model_path: str, precision: str):
    """
    Use the existing model_factory to wire up tokenizer/chat template/module
    pointers, but override _load_model so we control the dtype / quantization
    and skip the torch.compile wrap that breaks bnb / torchao quantized models.
    """
    family_cls = _resolve_family_class(model_path)
    orig_load = family_cls._load_model
    load_kwargs = build_quant_kwargs(precision)

    def patched_load(self, mp, dtype=None):
        # bf16/fp16 paths still go through this; we use the kwargs computed
        # for the requested precision rather than the family default.
        model = AutoModelForCausalLM.from_pretrained(mp, **load_kwargs).eval()
        model.requires_grad_(False)
        return model

    family_cls._load_model = patched_load
    try:
        base = construct_model_base(model_path)
    finally:
        family_cls._load_model = orig_load
    return base


def _resolve_family_class(model_path: str):
    """Return the ModelBase subclass that construct_model_base would pick."""
    mp = model_path.lower()
    if "qwen" in mp:
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel
    if "llama" in mp or "nemo" in mp:
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model
    if "gemma" in mp:
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel
    if "yi" in mp:
        from pipeline.model_utils.yi_model import YiModel
        return YiModel
    if "mistral" in mp:
        from pipeline.model_utils.mistral_model import MistralModel
        return MistralModel
    if "glm" in mp:
        from pipeline.model_utils.glm_model import GLM4Model
        return GLM4Model
    if "granite" in mp:
        from pipeline.model_utils.granite_hybrid import GraniteMoeHybridModel
        return GraniteMoeHybridModel
    if "phi" in mp:
        from pipeline.model_utils.phi_model import PhiModel
        return PhiModel
    raise ValueError(f"Unknown model family: {model_path}")


# ── Top-layer discovery ────────────────────────────────────────────────────────

def pick_top_layer(source_cfg: Config) -> Optional[int]:
    """Return the layer index with the highest full-direction ASR, or None.

    Reads from source_cfg's artifact tree (the original, non-quant runs dir)
    where sweep_layers.py's full_sweep results live.
    """
    full_dir = os.path.join(source_cfg.artifact_path(), "completions", "full_sweep")
    if not os.path.isdir(full_dir):
        return None
    import re
    rows = []
    for fn in os.listdir(full_dir):
        m = re.match(r"harmful_Full_layer(\d+)_evaluations\.json$", fn)
        if not m:
            continue
        with open(os.path.join(full_dir, fn)) as f:
            data = json.load(f)
        asr = data.get("llamaguard3_success_rate")
        if asr is not None:
            rows.append((int(m.group(1)), asr))
    if not rows:
        return None
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[0][0]


# ── Direction loading (no regeneration if cached) ─────────────────────────────

def load_or_build_directions(source_cfg: Config, model_path: str, batch_size: int):
    """Reuse cached cat_means.pt / harmless_reference.pt from the *source*
    runs/ tree (the non-quant alias). If not present, regenerate them there
    so subsequent quant runs can reuse them."""
    dir_path = os.path.join(source_cfg.artifact_path(), "generate_directions")
    cat_means_path = os.path.join(dir_path, "cat_means.pt")
    harmless_ref_path = os.path.join(dir_path, "harmless_reference.pt")

    if os.path.exists(cat_means_path) and os.path.exists(harmless_ref_path):
        print(f"  Reusing cached directions at {dir_path}")
        cat_means = torch.load(cat_means_path, weights_only=False)
        harmless_mean = torch.load(harmless_ref_path, weights_only=False)
        return cat_means, harmless_mean

    # Cache miss — generate. Needs a bf16 model in memory.
    print(f"  No cached directions in {dir_path}; generating from scratch...")
    base = construct_model_base_with_precision(model_path, "bf16")
    harmful_train = load_dataset_split(harmtype="mixed_harmful", split="train",
                                        instructions_only=False)
    harmless_train = load_dataset_split(harmtype="harmless", split="train",
                                         instructions_only=True)
    cat_means, harmless_mean = generate_and_save_category_directions(
        source_cfg, base, harmful_train, harmless_train,
        batch_size=batch_size, use_existing=False,
    )
    unload_model(base)
    return cat_means, harmless_mean


# ── Per-precision run ─────────────────────────────────────────────────────────

def run_one_precision(cfg, model_path, precision, layer_idx, direction,
                      harmful_test, batch_size, use_existing,
                      save_path="completions/quant_sweep"):
    """
    Load the model at the given precision, generate steered completions at the
    target layer. Returns the list of (tag, save_path) pairs to evaluate.
    """
    print(f"\n── Precision: {precision} ──────────────────────────────────────")
    base = construct_model_base_with_precision(model_path, precision)

    sorted_indices = direction.abs().argsort(descending=True).cpu().numpy()
    hook_fn = make_sparse_projection_input_hook(direction, sorted_indices.copy())
    steered_tag = f"Steered_{precision}_layer{layer_idx}"
    print(f"  Generating steered completions:  {steered_tag}")
    generate_and_save_completions_for_dataset(
        cfg, base,
        [(base.model_block_modules[layer_idx], hook_fn)], [],
        steered_tag, "harmful",
        batch_size=batch_size, dataset=harmful_test,
        save_path=save_path, use_existing=use_existing,
    )

    unload_model(base)
    return [(steered_tag, save_path)]


# ── Evaluation + summary ──────────────────────────────────────────────────────

def evaluate_all(cfg, tags_and_paths, batch_size, use_existing):
    print("\nEvaluating completions with LlamaGuard 3...")
    for tag, save_path in tags_and_paths:
        print(f"  {tag}")
        evaluate_completions_and_save_results_for_dataset(
            cfg, tag, "harmful",
            eval_methodologies=["llamaguard3"],
            save_path=save_path, use_existing=use_existing,
            batch_size=batch_size,
        )
    unload_llamaguard_model()
    gc.collect()
    torch.cuda.empty_cache()


def summarise(cfg, precisions, layer_idx, save_path="completions/quant_sweep"):
    """Read the per-precision steered evaluation JSONs and tabulate ASR."""
    art = cfg.artifact_path()
    rows = []
    for p in precisions:
        steer_eval = os.path.join(art, save_path,
                                   f"harmful_Steered_{p}_layer{layer_idx}_evaluations.json")
        steer_asr = None
        if os.path.exists(steer_eval):
            with open(steer_eval) as f:
                steer_asr = json.load(f).get("llamaguard3_success_rate")
        rows.append({
            "precision":   p,
            "steered_asr": steer_asr,
        })

    print("\n── Quant Steering Summary ──────────────────────────────────────────")
    print(f"  Layer: {layer_idx}")
    print(f"{'Precision':>10}  {'Steered ASR':>12}")
    print("─" * 28)
    for r in rows:
        s = f"{r['steered_asr']:.3f}" if r['steered_asr'] is not None else "   --"
        print(f"{r['precision']:>10}  {s:>12}")

    summary = {"layer": layer_idx, "rows": rows}
    summary_path = os.path.join(art, save_path, "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {summary_path}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--alias_suffix", type=str, default="-quant",
                        help="Suffix appended to model_alias for the output runs/ dir. "
                             "Default '-quant' writes to runs/{model}-quant/ while reading "
                             "directions from runs/{model}/. Set to '' to share the source dir.")
    parser.add_argument("--source_alias", type=str, default=None,
                        help="Alias to read directions / full_sweep results from. "
                             "Defaults to basename(model_path).")
    parser.add_argument("--precisions", type=str,
                        default="bf16,int8,nf4,fp4,fp8",
                        help=f"Comma-separated subset of {SUPPORTED_PRECISIONS}")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to steer at (defaults to top ASR layer from runs/.../full_sweep)")
    parser.add_argument("--num_test", type=int, default=128,
                        help="Number of harmful_test prompts to evaluate")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--use_existing", action="store_true",
                        help="Reuse cached completions/evaluations where present")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Generate completions only; skip LlamaGuard evaluation")
    parser.add_argument("--summarise_only", action="store_true",
                        help="Skip generation and evaluation; just aggregate "
                             "the per-precision evaluation JSONs into summary.json")
    args = parser.parse_args()

    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    for p in precisions:
        if p not in SUPPORTED_PRECISIONS:
            raise ValueError(f"Unsupported precision {p!r}. Supported: {SUPPORTED_PRECISIONS}")

    base_alias   = os.path.basename(args.model_path)
    source_alias = args.source_alias or base_alias
    output_alias = base_alias + args.alias_suffix

    source_cfg = Config(model_alias=source_alias, model_path=args.model_path)
    cfg        = Config(model_alias=output_alias, model_path=args.model_path)
    os.makedirs(cfg.artifact_path(), exist_ok=True)

    print(f"Source artifacts (read): {source_cfg.artifact_path()}")
    print(f"Output  artifacts (write): {cfg.artifact_path()}")

    # Fast path: aggregate existing per-precision evaluations and exit.
    if args.summarise_only:
        layer_idx = args.layer if args.layer is not None else pick_top_layer(source_cfg)
        if layer_idx is None:
            raise RuntimeError("--summarise_only requires --layer or a populated full_sweep dir.")
        summarise(cfg, precisions, layer_idx)
        print("\nDone (summarise-only).")
        return

    harmful_test = load_dataset_split(harmtype="mixed_harmful", split="test",
                                       instructions_only=False)[:args.num_test]
    print(f"Loaded {len(harmful_test)} harmful test prompts.")

    # Directions — read (and if needed, build) under the source alias
    print("\nLoading refusal directions...")
    cat_means, harmless_mean = load_or_build_directions(source_cfg, args.model_path, args.batch_size)
    first_direction = next(iter(cat_means.values()))[1]
    num_layers = first_direction.shape[1]
    directions = build_directions(cat_means, num_layers)

    # Pick the layer (also read from source_cfg)
    layer_idx = args.layer if args.layer is not None else pick_top_layer(source_cfg)
    if layer_idx is None:
        raise RuntimeError(
            "Could not determine top-ASR layer from runs/.../full_sweep. "
            "Pass --layer explicitly."
        )
    if layer_idx not in directions:
        raise RuntimeError(
            f"Layer {layer_idx} has no cached direction (available: {sorted(directions)[:5]}…)"
        )
    print(f"Steering layer: {layer_idx}")
    direction = directions[layer_idx]

    # Generate completions for each precision
    all_tags = []
    for precision in precisions:
        try:
            tags = run_one_precision(
                cfg, args.model_path, precision, layer_idx, direction,
                harmful_test, args.batch_size, args.use_existing,
            )
            all_tags.extend(tags)
        except Exception as e:
            print(f"  [!] Precision {precision} failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

    # Evaluate
    if not args.skip_eval:
        evaluate_all(cfg, all_tags, args.eval_batch_size, args.use_existing)
        summarise(cfg, precisions, layer_idx)
    else:
        print("\nSkipping evaluation (--skip_eval).")

    print("\nDone.")


if __name__ == "__main__":
    main()
