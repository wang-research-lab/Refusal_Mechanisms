"""Check how often Stage-2 greedy candidates tie within epsilon=0.005.

This tells us whether random tie-break (Exp 3) would actually diverge from
deterministic. If tie-set size = 1 at most steps, all seeds collapse to the
same mechanism and Exp 3 is uninformative regardless of how many seeds we run.

For each model, walks the deterministic alpha=0.95 trajectory and reports:
  - tie_size per step (number of candidates within epsilon of min drop)
  - whether the chosen removal had ties (would random tie-break have fired?)
"""
from __future__ import annotations

import os
from pipeline.extract_completed import (
    find_baseline_full_sets, index_remove_files, RUNS_ROOT, MODELS,
)

EPSILON = 0.005
TAU = 0.05


def trace_with_ties(pruning_dir, stage1_set, anchor, removal_index=None):
    bf = find_baseline_full_sets(pruning_dir, anchor)
    baseline_asr = None
    for ts, sz, a, _ in bf:
        if set(ts) == set(stage1_set):
            baseline_asr = a; break
    if baseline_asr is None:
        return None
    if removal_index is None:
        removal_index = index_remove_files(pruning_dir, anchor)
    current = set(stage1_set)
    cumulative_drop = 0.0
    steps = []
    while len(current) > 1 and cumulative_drop <= TAU:
        cands = []
        for name in current:
            key = (frozenset(current - {name}), name)
            if key in removal_index:
                cands.append((name, baseline_asr - removal_index[key]))
        if not cands:
            break
        min_drop = min(c[1] for c in cands)
        tied = [c for c in cands if abs(c[1] - min_drop) <= EPSILON]
        cands.sort(key=lambda c: (c[1], c[0]))
        chosen, drop = cands[0]
        steps.append({
            "step": len(steps), "set_size_before": len(current),
            "n_candidates": len(cands), "min_drop": min_drop,
            "tie_size": len(tied), "chosen": chosen, "chosen_drop": drop,
            "tied_with": sorted(n for n, _ in tied if n != chosen),
        })
        current = current - {chosen}
        cumulative_drop = drop
    return steps


def main():
    print("## Tie-density on alpha=0.95 deterministic trajectory (epsilon=0.005, up to tau=0.05)\n")
    print("| model | steps | steps_with_ties | total_ties | mean_tie_size | tie_step_fraction |")
    print("|-|-|-|-|-|-|")
    for alias, anchor in MODELS:
        d = os.path.join(RUNS_ROOT, alias, "completions", "pruning")
        if not os.path.isdir(d):
            continue
        bf = find_baseline_full_sets(d, anchor)
        sizes = sorted(set(sz for _, sz, _, _ in bf))
        if not sizes:
            continue
        # alpha=0.95 = middle size when 3 alphas are present, else max
        if len(sizes) >= 3:
            target_size = sizes[1]
        else:
            target_size = sizes[-1]
        stage1 = next(ts for ts, sz, _, _ in bf if sz == target_size)
        ri = index_remove_files(d, anchor)
        steps = trace_with_ties(d, stage1, anchor, ri)
        if not steps:
            print(f"| {alias} | (no data) | | | | |")
            continue
        with_ties = sum(1 for s in steps if s["tie_size"] > 1)
        total_ties = sum(s["tie_size"] - 1 for s in steps if s["tie_size"] > 1)
        mean_tie = sum(s["tie_size"] for s in steps) / len(steps)
        print(f"| {alias} | {len(steps)} | {with_ties} | {total_ties} "
              f"| {mean_tie:.2f} | {with_ties/len(steps):.0%} |")

    # Per-model detail (which steps had ties)
    for alias, anchor in MODELS:
        d = os.path.join(RUNS_ROOT, alias, "completions", "pruning")
        if not os.path.isdir(d):
            continue
        bf = find_baseline_full_sets(d, anchor)
        sizes = sorted(set(sz for _, sz, _, _ in bf))
        if not sizes:
            continue
        target_size = sizes[1] if len(sizes) >= 3 else sizes[-1]
        stage1 = next(ts for ts, sz, _, _ in bf if sz == target_size)
        ri = index_remove_files(d, anchor)
        steps = trace_with_ties(d, stage1, anchor, ri)
        if not steps:
            continue
        tied_steps = [s for s in steps if s["tie_size"] > 1]
        if tied_steps:
            print(f"\n### {alias} — steps with ties (random tie-break would have fired):")
            for s in tied_steps:
                print(f"  step {s['step']} (set_size={s['set_size_before']}): "
                      f"chose {s['chosen']} (drop={s['chosen_drop']:.4f}), "
                      f"tied with {s['tied_with']}")
        else:
            print(f"\n### {alias} — no ties; random tie-break never fires; all seeds = deterministic")


if __name__ == "__main__":
    main()
