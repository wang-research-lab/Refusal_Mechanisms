"""Replay a random-tie-break Stage 2 trajectory using cached candidate evals.

Mirrors stage2_trajectory's tie_break='random' branch with the same RNG seed
so we can recover the seed's identified mechanism from the cache without
running the model.
"""
import os
import json
import numpy as np
from pipeline.extract_completed import (
    find_baseline_full_sets, index_remove_files, RUNS_ROOT, MODELS, asr,
)

EPSILON = 0.005


def replay_random(pruning_dir, stage1_set, anchor, seed, tau=0.05, removal_index=None):
    bf = find_baseline_full_sets(pruning_dir, anchor)
    baseline_asr = None
    for ts, sz, a, _ in bf:
        if set(ts) == set(stage1_set):
            baseline_asr = a; break
    if baseline_asr is None:
        return None, None, "no baseline_full"
    if removal_index is None:
        removal_index = index_remove_files(pruning_dir, anchor)
    rng = np.random.default_rng(seed)
    current = set(stage1_set)
    history = [(frozenset(current), 0.0, baseline_asr)]
    while len(current) > 1:
        cands = []
        for name in current:
            key = (frozenset(current - {name}), name)
            if key in removal_index:
                cands.append((name, baseline_asr - removal_index[key]))
        if not cands:
            return list(current), baseline_asr, "cache exhausted"
        min_drop = min(c[1] for c in cands)
        tied = [c for c in cands if abs(c[1] - min_drop) <= EPSILON]
        if len(tied) > 1:
            chosen, drop = tied[int(rng.integers(len(tied)))]
        else:
            chosen, drop = tied[0]
        if drop > tau:
            return list(current), baseline_asr - drop, None
        new_set = current - {chosen}
        history.append((frozenset(new_set), drop, baseline_asr - drop))
        current = new_set
    return list(current), baseline_asr, None


def main():
    print("## Random-seed mechanisms vs deterministic (alpha=0.95, tau=0.05)\n")
    for alias, anchor in MODELS:
        d = os.path.join(RUNS_ROOT, alias, "completions", "pruning")
        if not os.path.isdir(d):
            continue
        bf = find_baseline_full_sets(d, anchor)
        sizes = sorted(set(sz for _, sz, _, _ in bf))
        if len(sizes) < 2:
            continue
        # alpha=0.95 stage 1 is the middle size when 3 alphas present, else max
        target_size = sizes[1] if len(sizes) >= 3 else sizes[-1]
        stage1 = next(ts for ts, sz, _, _ in bf if sz == target_size)
        ri = index_remove_files(d, anchor)

        # Deterministic replay (sorted by (drop, name))
        from pipeline.extract_completed import replay_greedy
        det_mechs = replay_greedy(d, stage1, [0.05], anchor, ri)
        det_set = set(det_mechs[0.05][0])
        print(f"### {alias}")
        print(f"  deterministic mechanism (|M|={len(det_set)}): {sorted(det_set)}")

        # Each random seed
        for seed in (0, 1, 2):
            mech, val_asr, note = replay_random(d, stage1, anchor, seed, removal_index=ri)
            if mech is None:
                continue
            seed_set = set(mech)
            inter = det_set & seed_set
            union = det_set | seed_set
            jac = len(inter) / len(union) if union else 1.0
            is_subset = seed_set == det_set
            mark = "IDENTICAL" if is_subset else f"differs (Jaccard {jac:.2f})"
            note_str = f" [{note}]" if note else ""
            # Try to read the seed's test_asr if it was actually run
            tag = f"_exp3_s{seed}_test_evaluations.json"
            import glob
            te = glob.glob(os.path.join(d, f"*{tag}"))
            ta = asr(te[0]) if te else None
            ta_str = f", test_asr={ta:.3f}" if ta is not None else ""
            print(f"  seed={seed}: |M|={len(seed_set)}, vs det: {mark}{ta_str}{note_str}")
            if seed_set != det_set:
                added = seed_set - det_set
                removed = det_set - seed_set
                if added:
                    print(f"           in seed not det: {sorted(added)}")
                if removed:
                    print(f"           in det not seed: {sorted(removed)}")
        print()


if __name__ == "__main__":
    main()
