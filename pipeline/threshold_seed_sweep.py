"""Threshold sweep + seed stability for refusal-mechanism identification (rebuttal).

Builds directly on pipeline/steer_components.py:
  Stage 1 = heuristic_elimination_search  (projection elimination at alpha; deterministic)
  Stage 2 = greedy ASR elimination at tau  (here: stage2_trajectory, which logs the
            full elimination trajectory instead of stopping at a single tau)

Three experiments (see the rebuttal spec):
  Exp 1  tau-sweep at fixed alpha=0.95  — one extended Stage-2 trajectory per model,
         then tau in {0.02, 0.05, 0.10} are read off as stopping points on it.
  Exp 2  alpha-sweep at fixed tau=0.05  — alpha in {0.90, 0.99} need fresh Stage 1+2;
         alpha=0.95 is loaded from the paper's existing run.
  Exp 3  seed stability at alpha=0.95, tau=0.05 — Stage 2 with random tie-breaking
         (epsilon=0.005) over seeds {0,1,2}, sharing the cached Stage-1 output.

The Stage-2 "drop" is the *cumulative* ASR drop from the Stage-1-set baseline (this
mirrors iterative_pruning, where baseline_tag is fixed). The greedy min-drop ordering
is therefore a single trajectory and tau is only a stopping point on it — this is what
makes the Exp-1 one-trajectory trick valid.

NOTE ON COST: like iterative_pruning, each step regenerates completions for every
candidate removal over the val set and reloads the model between steps. This is the
existing pipeline's behaviour, kept for fidelity. Use --n_val to subsample for smoke
tests / paper reproduction before committing to full runs.

Usage:
  python -m pipeline.threshold_seed_sweep --model_path google/gemma-2-2b-it \
         --anchor_layer 13 --experiment 1
  python -m pipeline.threshold_seed_sweep ... --experiment all --report
"""
from __future__ import annotations

import argparse
import gc
import json
import os

import numpy as np
import torch

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from dataset.load_dataset import load_dataset_split
from pipeline.steer_components import (
    load_directions,
    build_directions,
    heuristic_elimination_search,
    rebuild_module_set,
    run_component_test,
    evaluate_tags,
    get_asr_from_evaluation_file,
    unload_model,
)

ALPHA_PAPER = 0.95
TAU_PAPER = 0.05
TAU_SWEEP = [0.02, 0.05, 0.10]
ALPHA_SWEEP = [0.90, 0.95, 0.99]
SEEDS = [0, 1, 2]
EPSILON = 0.005

OUT_NAME = "rebuttal_threshold_seed_results.json"   # combined report (one file, all models)
SIDECAR_NAME = "threshold_seed_sidecar.json"        # per-model, written to its own artifact dir
DEFAULT_RUNS_ROOT = "pipeline/runs"


# ── model / data / direction context (mirrors steer_components.main) ─────────────

class Ctx:
    """Holds the fixed per-model context: model_base, directions, data splits."""
    def __init__(self, model_path, anchor_layer, batch_size=128, n_test=512, n_val=None):
        self.model_path = model_path
        self.anchor_layer = anchor_layer
        self.batch_size = batch_size
        self.cfg = Config(model_alias=os.path.basename(model_path), model_path=model_path)
        self.model_base = construct_model_base(model_path)

        harmful_train = load_dataset_split("mixed_harmful", "train", instructions_only=False)
        harmless_train = load_dataset_split("harmless", "train", instructions_only=True)[:len(harmful_train)]
        harmful_val = load_dataset_split("mixed_harmful", "val", instructions_only=False)
        harmful_test = load_dataset_split("mixed_harmful", "test", instructions_only=False)[:n_test]
        if n_val:                       # smoke-test subsample
            harmful_val = harmful_val[:n_val]
        self.harmful_train, self.harmless_train = harmful_train, harmless_train
        self.harmful_val, self.harmful_test = harmful_val, harmful_test

        category_directions, harmless_mean = load_directions(
            self.cfg, self.model_base, harmful_train, harmless_train,
            batch_size=batch_size, use_existing=True,
        )
        num_layers = next(iter(category_directions.values()))[1].shape[1]
        self.directions = build_directions(self.cfg, category_directions, num_layers)
        self.harmless_mean = harmless_mean[-1]

    def reload(self):
        self.model_base = construct_model_base(self.model_path)
        return self.model_base

    @property
    def total_upstream(self):
        # embed + attn_0..L-1 + mlp_0..L-1  == 2*L + 1
        return 2 * self.anchor_layer + 1

    def full_upstream_names(self):
        names = ["embed"]
        for i in range(self.anchor_layer):
            names += [f"attn_{i}", f"mlp_{i}"]
        return names


# ── ASR of an arbitrary component set (used for test ASR + full-steering ref) ────

def set_asr(ctx: Ctx, names, prompts, tag, use_existing=True):
    """Generate completions steering exactly `names`, evaluate, return ASR.

    Mirrors the circuit-performance test in steer_components.main (run_component_test
    on the test set -> evaluate_tags -> llamaguard3_success_rate).
    """
    model_base = ctx.reload()
    circuit = rebuild_module_set(list(names), model_base, ctx.directions, ctx.anchor_layer)
    # ablation must be truthy so completions/evals land in completions/pruning, where
    # get_asr_from_evaluation_file reads from. override_tag still wins for the tag name,
    # so "ablate_*" is not appended — only the save path is routed to pruning.
    tags = run_component_test(
        ctx.cfg, model_base, circuit, ctx.directions,
        override_tag=tag, ablation="circuit_test", anchor_layer=ctx.anchor_layer,
        harmful_prompts=prompts, batch_size=ctx.batch_size, use_existing=use_existing,
    )
    unload_model(model_base); gc.collect(); torch.cuda.empty_cache()
    evaluate_tags(ctx.cfg, tags, batch_size=ctx.batch_size, use_existing=use_existing)
    return get_asr_from_evaluation_file(ctx.cfg, os.path.basename(tags[0][0]))


# ── Stage 2 with full trajectory logging + configurable tie-break ────────────────

def stage2_trajectory(ctx: Ctx, stage1_names, *, max_tau=0.10,
                      tie_break="lowest_index", seed=None, epsilon=EPSILON,
                      use_existing=True):
    """Greedy ASR elimination from `stage1_names`, logging every step.

    Removes the min-cumulative-drop component each step and keeps going while that
    drop <= max_tau, so all tau <= max_tau stopping points live on the one trajectory.

    tie_break:
      "lowest_index" — deterministic; among min-drop candidates take the lowest
                       (drop, name) — reproduces iterative_pruning's min() ordering.
      "random"       — among candidates within `epsilon` of the min drop, choose one
                       uniformly using `seed`. Records when this was actually invoked.

    Tags use the SAME content-addressed scheme as iterative_pruning ("baseline_full",
    "remove_{name}") — run_component_test encodes the full steered set into the tag, so
    with use_existing=True the deterministic alpha=0.95 trajectory reuses the paper's
    cached completions step-for-step (only the tau>0.05 extension generates fresh), and
    any (set, removal) shared across experiments/seeds is reused too.

    Returns (trajectory, baseline_asr). trajectory entries:
      {step, removed, drop, new_asr, baseline_asr, set_before, tie_size, random_choice}
    """
    rng = np.random.default_rng(seed) if tie_break == "random" else None
    cfg, anchor, bs = ctx.cfg, ctx.anchor_layer, ctx.batch_size

    model_base = ctx.reload()
    current = rebuild_module_set(list(stage1_names), model_base, ctx.directions, anchor)

    # Fixed baseline = ASR of the full Stage-1 set on val (matches iterative_pruning).
    baseline_tag = run_component_test(
        cfg, model_base, current, ctx.directions, ablation="baseline_full",
        anchor_layer=anchor, harmful_prompts=ctx.harmful_val, batch_size=bs, use_existing=use_existing,
    )
    unload_model(model_base); gc.collect(); torch.cuda.empty_cache()
    evaluate_tags(cfg, baseline_tag, batch_size=bs, use_existing=use_existing)
    baseline_asr = get_asr_from_evaluation_file(cfg, os.path.basename(baseline_tag[0][0]))

    model_base = ctx.reload()
    current = rebuild_module_set([n for n, _, _ in current], model_base, ctx.directions, anchor)

    trajectory = []
    step = 0
    while len(current) > 1:
        # one generation per candidate removal; gen_tags[i] <-> removing current[i].
        # ablation="remove_{name}" + the test_set abbrev make this tag content-addressed,
        # so identical (set, removal) computations reuse cached completions.
        gen_tags = []
        for i, (name, _, _) in enumerate(current):
            test_set = current[:i] + current[i + 1:]
            tags = run_component_test(
                cfg, model_base, test_set, ctx.directions,
                ablation=f"remove_{name}", anchor_layer=anchor,
                harmful_prompts=ctx.harmful_val, batch_size=bs, use_existing=use_existing,
            )
            gen_tags.extend(tags)

        unload_model(model_base); gc.collect(); torch.cuda.empty_cache()
        evaluate_tags(cfg, baseline_tag + gen_tags, batch_size=bs, use_existing=use_existing)
        baseline_asr = get_asr_from_evaluation_file(cfg, os.path.basename(baseline_tag[0][0]))

        # cands: (name, cumulative_drop, asr, index_in_current)
        cands = []
        for i, (tagname, _) in enumerate(gen_tags):
            asr = get_asr_from_evaluation_file(cfg, tagname)
            cands.append((current[i][0], baseline_asr - asr, asr, i))

        choice, tie_size, random_invoked = _choose(cands, tie_break, rng, epsilon)
        removed, drop, new_asr, idx = choice
        trajectory.append({
            "step": step, "removed": removed, "drop": float(drop), "new_asr": float(new_asr),
            "baseline_asr": float(baseline_asr), "set_before": [n for n, _, _ in current],
            "tie_size": tie_size, "random_choice": random_invoked,
        })

        if drop > max_tau:
            break  # passed the largest tau of interest; trajectory complete
        current.pop(idx)
        model_base = ctx.reload()
        current = rebuild_module_set([n for n, _, _ in current], ctx.model_base, ctx.directions, anchor)
        step += 1

    return trajectory, baseline_asr


def _choose(cands, tie_break, rng, epsilon):
    """Pick a removal candidate. Returns (chosen, tie_size, random_invoked)."""
    min_drop = min(c[1] for c in cands)
    if tie_break == "random":
        tied = [c for c in cands if abs(c[1] - min_drop) <= epsilon]
        if len(tied) > 1:
            return tied[int(rng.integers(len(tied)))], len(tied), True
        return tied[0], 1, False
    # deterministic: smallest (drop, name) — stable, lowest-index-style
    chosen = min(cands, key=lambda c: (c[1], c[0]))
    exact_ties = sum(1 for c in cands if abs(c[1] - min_drop) <= 1e-9)
    return chosen, exact_ties, False


# ── pure-logic helpers (unit-testable without a model) ───────────────────────────

def extract_mechanism_at_tau(trajectory, stage1_names, tau):
    """Mechanism = set just before the first step whose cumulative drop exceeds tau.

    Returns (mechanism_names, val_asr, note). Walks the trajectory in order, removing
    while drop <= tau, stopping at the first drop > tau. The mechanism's val ASR is
    recoverable from the trajectory: the ASR of set S_k is the new_asr recorded for
    the accepted removal that produced it (baseline_asr for the full Stage-1 set).
    """
    if not trajectory:
        return list(stage1_names), None, "empty trajectory (Stage-1 set already minimal)"
    for k, entry in enumerate(trajectory):
        if entry["drop"] > tau:
            val_asr = entry["baseline_asr"] if k == 0 else trajectory[k - 1]["new_asr"]
            return list(entry["set_before"]), val_asr, None
    # never exceeded tau -> trajectory exhausted; mechanism = final set after last removal
    last = trajectory[-1]
    final = [n for n in last["set_before"] if n != last["removed"]]
    return final, last["new_asr"], f"trajectory exhausted before drop exceeded tau={tau}; reporting final set"


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def aggregate_seeds(seed_records):
    """seed_records: {seed_key: {mechanism_components, test_asr, ...}}. seed_key
    may be an int (random seed) or a string (e.g. "paper_det"). Returns aggregate dict."""
    keys = sorted(seed_records, key=str)
    sets = {s: set(seed_records[s]["mechanism_components"]) for s in keys}
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    pairwise = {f"{a}_vs_{b}": jaccard(sets[a], sets[b]) for a, b in pairs}
    sizes = [len(sets[s]) for s in keys]
    test_asrs = [seed_records[s]["test_asr"] for s in keys]
    inter = set.intersection(*sets.values()) if sets else set()
    union = set.union(*sets.values()) if sets else set()
    return {
        "pairwise_jaccard": pairwise,
        "mean_jaccard": float(np.mean(list(pairwise.values()))) if pairwise else 1.0,
        "mechanism_size_range": [min(sizes), max(sizes)],
        "test_asr_range": [min(test_asrs), max(test_asrs)],
        "core_components": sorted(inter),
        "core_size": len(inter),
        "peripheral_components": sorted(union - inter),
        "peripheral_count": len(union - inter),
        "n_realizations": len(keys),
        "realization_keys": [str(k) for k in keys],
    }


def mechanism_record(ctx, names, *, alpha, tau, val_asr, test_asr, full_asr, extra=None):
    total = ctx.total_upstream
    rec = {
        "model_name": ctx.model_path,
        "alpha": alpha,
        "tau": tau,
        "mechanism_components": sorted(names),
        "mechanism_size": len(names),
        "mechanism_fraction": len(names) / total,
        "validation_asr": val_asr,
        "test_asr": test_asr,
        "full_steering_asr": full_asr,
    }
    if extra:
        rec.update(extra)
    return rec


# ── experiments ──────────────────────────────────────────────────────────────────

def stage1(ctx, alpha):
    """Run (or load) deterministic Stage 1 at the given alpha; return component names.

    Reloads the model first: set_asr / stage2_trajectory leave it unloaded, and even
    Stage 1's use_existing path rebuilds live module refs (model.embed_tokens).
    """
    model_base = ctx.reload()
    circuit = heuristic_elimination_search(
        ctx.cfg, model_base, ctx.directions, ctx.harmful_val, ctx.harmless_mean,
        heuristic_threshold=alpha, anchor_layer=ctx.anchor_layer,
        batch_size=ctx.batch_size, use_existing=True, temperature=None,
    )
    return [c[0] for c in circuit]


def run_exp1(ctx, full_asr, *, save_cb=None):
    """tau-sweep at alpha=0.95 from one extended deterministic trajectory.

    save_cb(out) is called after each tau's test ASR completes so the sidecar
    captures partial progress.
    """
    s1 = stage1(ctx, ALPHA_PAPER)
    traj, _ = stage2_trajectory(ctx, s1, max_tau=max(TAU_SWEEP),
                                tie_break="lowest_index")
    out = {"_trajectory": traj}
    for tau in TAU_SWEEP:
        mech, val_asr, note = extract_mechanism_at_tau(traj, s1, tau)
        test_asr = set_asr(ctx, mech, ctx.harmful_test, tag=f"_exp1_a95_t{tau}_test")
        rec = mechanism_record(ctx, mech, alpha=ALPHA_PAPER, tau=tau,
                               val_asr=val_asr, test_asr=test_asr, full_asr=full_asr,
                               extra={"note": note} if note else None)
        out[f"tau_{tau}"] = rec
        if save_cb: save_cb(out)
    return out


def run_exp2(ctx, full_asr, exp1_block=None, *, only_alphas=None, save_cb=None):
    """alpha-sweep at tau=0.05. alpha=0.95 reuses Exp-1's tau_0.05 result if given.

    only_alphas: iterable of floats; if set, only these alphas run.
    save_cb(out) fires after each alpha completes.
    """
    alphas = ALPHA_SWEEP if only_alphas is None else [a for a in ALPHA_SWEEP if a in only_alphas]
    out = {}
    for alpha in alphas:
        if alpha == ALPHA_PAPER and exp1_block and "tau_0.05" in exp1_block:
            out[f"alpha_{alpha}"] = dict(exp1_block["tau_0.05"])  # paper result, already computed
            if save_cb: save_cb(out)
            continue
        s1 = stage1(ctx, alpha)
        traj, _ = stage2_trajectory(ctx, s1, max_tau=TAU_PAPER,
                                    tie_break="lowest_index")
        mech, val_asr, note = extract_mechanism_at_tau(traj, s1, TAU_PAPER)
        test_asr = set_asr(ctx, mech, ctx.harmful_test, tag=f"_exp2_a{alpha}_t05_test")
        out[f"alpha_{alpha}"] = mechanism_record(
            ctx, mech, alpha=alpha, tau=TAU_PAPER,
            val_asr=val_asr, test_asr=test_asr, full_asr=full_asr,
            extra={"note": note} if note else None)
        if save_cb: save_cb(out)
    return out


def run_exp3(ctx, full_asr, *, only_seeds=None, save_cb=None):
    """seed stability at alpha=0.95, tau=0.05 with random tie-breaking.

    only_seeds: iterable of ints; if set, only these seeds run.
    save_cb(out) fires after each seed completes. Aggregate is recomputed each
    time over whatever seeds are present.
    """
    seeds = SEEDS if only_seeds is None else [s for s in SEEDS if s in only_seeds]
    s1 = stage1(ctx, ALPHA_PAPER)        # cached deterministic Stage-1 output (shared)
    out = {}
    for seed in seeds:
        traj, _ = stage2_trajectory(ctx, s1, max_tau=TAU_PAPER, tie_break="random",
                                    seed=seed, epsilon=EPSILON)
        mech, val_asr, note = extract_mechanism_at_tau(traj, s1, TAU_PAPER)
        test_asr = set_asr(ctx, mech, ctx.harmful_test, tag=f"_exp3_s{seed}_test")
        rec = mechanism_record(ctx, mech, alpha=ALPHA_PAPER, tau=TAU_PAPER,
                               val_asr=val_asr, test_asr=test_asr, full_asr=full_asr,
                               extra={"seed": seed, "epsilon": EPSILON,
                                      "num_random_choices": sum(e["random_choice"] for e in traj)})
        if note:
            rec["note"] = note
        out[f"seed_{seed}"] = rec
        seed_records = {s: out[f"seed_{s}"] for s in seeds if f"seed_{s}" in out}
        if len(seed_records) >= 2:
            out["aggregate"] = aggregate_seeds(seed_records)
        if save_cb: save_cb(out)
    return out


# ── reporting ─────────────────────────────────────────────────────────────────────

def markdown_tables(summary):
    """Two compact markdown tables for the rebuttal text."""
    e1, e2, e3 = (summary.get("experiment_1_tau_sweep", {}),
                  summary.get("experiment_2_alpha_sweep", {}),
                  summary.get("experiment_3_seed_stability", {}))
    models = sorted(set(e1) | set(e2) | set(e3))

    def cell(rec):
        if not rec:
            return "— / —"
        return f"{rec['mechanism_size']} ({rec['mechanism_fraction']:.0%}) / {rec['test_asr']:.2f}"

    t1 = ["### Table 1 — Threshold sensitivity (mechanism size (frac) / test ASR)",
          "| model | α0.90,τ0.05 | α0.95,τ0.02 | α0.95,τ0.05 | α0.95,τ0.10 | α0.99,τ0.05 |",
          "|---|---|---|---|---|---|"]
    for m in models:
        a = e1.get(m, {}); b = e2.get(m, {})
        row = [m,
               cell(b.get("alpha_0.9") or b.get("alpha_0.90")),
               cell(a.get("tau_0.02")),
               cell(a.get("tau_0.05")),
               cell(a.get("tau_0.1") or a.get("tau_0.10")),
               cell(b.get("alpha_0.99"))]
        t1.append("| " + " | ".join(row) + " |")

    t2 = ["", "### Table 2 — Seed stability (α=0.95, τ=0.05)",
          "| model | mean Jaccard | size range | test ASR range | core size | peripheral |",
          "|---|---|---|---|---|---|"]
    for m in models:
        agg = e3.get(m, {}).get("aggregate")
        if not agg:
            t2.append(f"| {m} | — | — | — | — | — |"); continue
        sr = agg["mechanism_size_range"]; ar = agg["test_asr_range"]
        t2.append(f"| {m} | {agg['mean_jaccard']:.3f} | {sr[0]}–{sr[1]} | "
                  f"{ar[0]:.2f}–{ar[1]:.2f} | {agg['core_size']} | {agg['peripheral_count']} |")
    return "\n".join(t1 + t2)


# ── validation checks (spec) ───────────────────────────────────────────────────────

def validation_checks(model_block):
    """Return list of (severity, message) flags per the spec's validation section."""
    flags = []
    e1 = model_block.get("exp1", {})
    e2 = model_block.get("exp2", {})
    full = None
    # tau monotonicity: larger tau -> smaller (or equal) mechanism
    sizes = [(t, e1.get(f"tau_{t}", {}).get("mechanism_size")) for t in TAU_SWEEP]
    sizes = [(t, s) for t, s in sizes if s is not None]
    for (t0, s0), (t1, s1) in zip(sizes, sizes[1:]):
        if s1 > s0:
            flags.append(("ERROR", f"tau monotonicity broken: tau={t1} size {s1} > tau={t0} size {s0}"))
    # test ASR >= 70% of full steering
    for block in (e1, e2):
        for key, rec in block.items():
            if not isinstance(rec, dict) or "test_asr" not in rec:
                continue
            full = rec.get("full_steering_asr") or full
            if full and rec["test_asr"] < 0.70 * full:
                flags.append(("WARN", f"{key}: test ASR {rec['test_asr']:.2f} < 70% of full {full:.2f}"))
            frac = rec.get("mechanism_fraction")
            if frac is not None and not (0.20 <= frac <= 0.60):
                flags.append(("FLAG", f"{key}: mechanism fraction {frac:.0%} outside [20%,60%]"))
    # seed jaccard
    agg = model_block.get("exp3", {}).get("aggregate", {})
    if agg and agg.get("mean_jaccard", 1.0) < 0.5:
        flags.append(("FLAG", f"mean Jaccard {agg['mean_jaccard']:.2f} < 0.5 (high seed variance)"))
    return flags


# ── aggregation: per-model sidecars -> combined report ───────────────────────────

def _merge_block(dst, src):
    """Deep-merge src into dst at the exp1/exp2/exp3 key level.

    For exp1/exp2 (dicts keyed by tau_*/alpha_*), per-key records from src
    overwrite dst's (last shard wins on collision — fine since records are
    derivable from the same content-addressed cache).
    For exp3, per-seed records merge and the aggregate is recomputed after.
    """
    for k in ("exp1", "exp2", "exp3"):
        s = src.get(k)
        if not s:
            continue
        dst.setdefault(k, {}).update(s)
    if "full_steering_asr" in src and src["full_steering_asr"] is not None:
        dst["full_steering_asr"] = src["full_steering_asr"]
    if "model" in src and "model" not in dst:
        dst["model"] = src["model"]
    return dst


def aggregate_report(runs_root, out_path, include_paper_det_as_seed=False):
    """Gather every model's threshold_seed_sidecar*.json shards into the combined
    summary JSON + markdown tables. Multiple shards per model are merged.

    include_paper_det_as_seed: if True and Exp 2 alpha=0.95 (or Exp 1 tau=0.05)
    mechanism is present, include it as an additional "seed_paper" record in the
    Exp 3 aggregate. Lets us count the paper's deterministic mechanism as the
    third realization when only 2 random seeds were run.
    """
    import glob
    summary = {"experiment_1_tau_sweep": {}, "experiment_2_alpha_sweep": {},
               "experiment_3_seed_stability": {}}
    flags = {}
    shard_paths = sorted(glob.glob(os.path.join(runs_root, "*", "threshold_seed_sidecar*.json")))
    by_alias = {}
    for sc in shard_paths:
        try:
            block = json.load(open(sc))
        except json.JSONDecodeError:
            print(f"  skip (invalid JSON): {sc}")
            continue
        alias = block.get("model") or os.path.basename(os.path.dirname(sc))
        by_alias.setdefault(alias, []).append((sc, block))

    for alias, shards in by_alias.items():
        merged = {"model": alias}
        for sc, block in shards:
            _merge_block(merged, block)
        e3 = merged.get("exp3", {}) or {}
        seed_records = {int(k.split("_")[1]): v for k, v in e3.items()
                        if k.startswith("seed_") and isinstance(v, dict)}
        if include_paper_det_as_seed:
            det_mech = (merged.get("exp2", {}).get(f"alpha_{ALPHA_PAPER}")
                        or merged.get("exp1", {}).get("tau_0.05"))
            if det_mech and "mechanism_components" in det_mech:
                seed_records["paper_det"] = det_mech
        if len(seed_records) >= 2:
            e3["aggregate"] = aggregate_seeds(seed_records)
        merged["exp3"] = e3
        summary["experiment_1_tau_sweep"][alias] = merged.get("exp1", {})
        summary["experiment_2_alpha_sweep"][alias] = merged.get("exp2", {})
        summary["experiment_3_seed_stability"][alias] = merged.get("exp3", {})
        merged["validation_flags"] = [
            {"severity": s, "message": m} for s, m in validation_checks(merged)
        ]
        if merged["validation_flags"]:
            flags[alias] = merged["validation_flags"]

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_path.replace(".json", ".md"), "w") as f:
        f.write(markdown_tables(summary))
    print(f"Aggregated {len(shard_paths)} shard(s) across {len(by_alias)} model(s) -> {out_path}  (+ .md tables)")
    for alias, fl in flags.items():
        for x in fl:
            print(f"[{alias}] [{x['severity']}] {x['message']}")
    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path")
    p.add_argument("--anchor_layer", type=int)
    p.add_argument("--experiment", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_test", type=int, default=512)
    p.add_argument("--n_val", type=int, default=None, help="subsample val for smoke tests")
    p.add_argument("--only-alpha", type=float, nargs="+", default=None,
                   dest="only_alpha",
                   help="restrict Exp 2 to these alphas (e.g. --only-alpha 0.99)")
    p.add_argument("--only-seed", type=int, nargs="+", default=None,
                   dest="only_seed",
                   help="restrict Exp 3 to these seeds (e.g. --only-seed 0 1)")
    p.add_argument("--sidecar-suffix", default="", dest="sidecar_suffix",
                   help="suffix on sidecar filename for parallel shards (e.g. _seed0)")
    p.add_argument("--report", action="store_true",
                   help="aggregate per-model sidecar shards into the combined summary + tables; runs no experiments")
    p.add_argument("--include-paper-det-as-seed", action="store_true",
                   dest="include_paper_det",
                   help="(report only) count Exp2 alpha=0.95 deterministic mechanism as a third realization in Exp 3 Jaccard")
    p.add_argument("--runs_root", default=DEFAULT_RUNS_ROOT,
                   help="dir holding per-model artifact dirs (for --report)")
    p.add_argument("--out", default=None, help=f"combined report path (default <runs_root>/{OUT_NAME})")
    args = p.parse_args()

    if args.report:
        aggregate_report(args.runs_root, args.out or os.path.join(args.runs_root, OUT_NAME),
                         include_paper_det_as_seed=args.include_paper_det)
        return

    if not args.model_path or args.anchor_layer is None:
        p.error("--model_path and --anchor_layer are required unless --report")

    ctx = Ctx(args.model_path, args.anchor_layer, batch_size=args.batch_size,
              n_test=args.n_test, n_val=args.n_val)
    alias = os.path.basename(args.model_path)

    # full-component steering reference ASR on the test set (the 2L+1 baseline)
    full_asr = set_asr(ctx, ctx.full_upstream_names(), ctx.harmful_test, tag="_full_steering_test")
    print(f"[{alias}] full-steering test ASR = {full_asr:.3f}")

    sidecar_name = SIDECAR_NAME.replace(".json", f"{args.sidecar_suffix}.json")
    sidecar = args.out or os.path.join(ctx.cfg.artifact_path(), sidecar_name)
    model_block = {"model": alias, "full_steering_asr": full_asr,
                   "_status": "starting"}

    def save():
        with open(sidecar, "w") as f:
            json.dump(model_block, f, indent=2)

    save()  # initial sidecar with full_steering_asr

    if args.experiment in ("1", "all"):
        model_block["_status"] = "exp1"
        def save_exp1(partial):
            model_block["exp1"] = partial; save()
        model_block["exp1"] = run_exp1(ctx, full_asr, save_cb=save_exp1)
    if args.experiment in ("2", "all"):
        model_block["_status"] = "exp2"
        def save_exp2(partial):
            model_block["exp2"] = partial; save()
        model_block["exp2"] = run_exp2(ctx, full_asr, model_block.get("exp1"),
                                       only_alphas=args.only_alpha, save_cb=save_exp2)
    if args.experiment in ("3", "all"):
        model_block["_status"] = "exp3"
        def save_exp3(partial):
            model_block["exp3"] = partial; save()
        model_block["exp3"] = run_exp3(ctx, full_asr,
                                       only_seeds=args.only_seed, save_cb=save_exp3)

    model_block["_status"] = "done"
    model_block["validation_flags"] = [
        {"severity": s, "message": m} for s, m in validation_checks(model_block)
    ]
    for x in model_block["validation_flags"]:
        print(f"[{alias}] [{x['severity']}] {x['message']}")
    save()
    print(f"[{alias}] wrote sidecar {sidecar}  (run with --report to build the combined tables)")


if __name__ == "__main__":
    main()
