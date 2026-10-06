"""Extract test ASR + mechanism size from completed milestones in the cache.

Replays the greedy Stage-2 trajectory against cached val evaluations to recover
each (alpha, tau) mechanism, then reads the test ASR from the test_*_evaluations
file directly. No model loading or fresh generation — pure read.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

RUNS_ROOT = "pipeline/runs"

# Stage 1 sizes at alpha=0.95 (anchored from the paper's exhaustive_set files).
# alpha=0.90 / 0.99 stage 1 sizes are recovered from the cache below.
MODELS = [
    ("gemma-2-2b-it", 13),
    ("Phi-4-mini-instruct", 14),
    ("Qwen3-4B", 19),
    ("Mistral-Small-3.2-24B-Instruct-2506", 17),
]


def parse_components(s):
    """Decode short-tag like 'a2m3a5e' into ['attn_2','mlp_3','attn_5','embed'].
    Per steer_components: mlp_3 -> m3, attn_5 -> a5, embed -> e."""
    out = []
    pat = re.compile(r"a(\d+)|m(\d+)|e(?![a-z])")
    for m in pat.finditer(s):
        if m.group(1) is not None:
            out.append("attn_" + m.group(1))
        elif m.group(2) is not None:
            out.append("mlp_" + m.group(2))
        else:
            out.append("embed")
    return out


def asr(eval_path):
    try:
        return json.load(open(eval_path)).get("llamaguard3_success_rate")
    except (json.JSONDecodeError, OSError):
        return None


def find_baseline_full_sets(pruning_dir, anchor):
    """Find all (set, baseline_asr) pairs from baseline_full evals.

    Each baseline_full file represents the val ASR of a Stage-1 set. We need
    these to identify which Stage-1 (alpha) produced which trajectory.
    Returns list of (sorted_set_tuple, set_size, baseline_asr, file_basename).
    """
    out = []
    for f in glob.glob(os.path.join(pruning_dir, f"*_L{anchor}_*ablate_baseline_full_evaluations.json")):
        name = os.path.basename(f)
        # tag is between L<anchor>_ and ablate_baseline_full
        m = re.search(rf"_L{anchor}_(.+?)ablate_baseline_full", name)
        if not m:
            continue
        comps = parse_components(m.group(1))
        a = asr(f)
        if a is None:
            continue
        out.append((tuple(sorted(comps)), len(comps), a, name))
    return out


def index_remove_files(pruning_dir, anchor):
    """Build {(frozenset(remaining_after), removed_name): asr} from all
    cached remove_* eval files. We match by component-SET (not tag order)
    because run_component_test encodes Stage-1 ordering which we don't have."""
    idx = {}
    pattern = os.path.join(pruning_dir, f"*_L{anchor}_*ablate_remove_*_evaluations.json")
    for f in glob.glob(pattern):
        name = os.path.basename(f)
        m = re.search(rf"_L{anchor}_(.+?)ablate_remove_(attn_\d+|mlp_\d+|embed)_evaluations",
                      name)
        if not m:
            continue
        before_tag, removed = m.group(1), m.group(2)
        remaining = parse_components(before_tag)
        a = asr(f)
        if a is None:
            continue
        # remaining = set BEFORE the removal; remove `removed` to get the candidate set tested
        # But careful: run_component_test names the file with the REMAINING set in the tag,
        # then "remove_X" means we ablate X FROM that remaining set. So the tested set =
        # remaining \ {removed}. The reference "current" in greedy is `remaining` though,
        # because we evaluate "what if we remove X from current?"
        idx[(frozenset(remaining), removed)] = a
    return idx


def replay_greedy(pruning_dir, stage1_set, taus, anchor, removal_index=None):
    """From the Stage-1 set, replay the greedy-min-drop trajectory against
    cached evals. Returns {tau: (mechanism_set, val_asr, n_steps)}.

    Mirrors stage2_trajectory's lowest_index tie-break.
    """
    bf = find_baseline_full_sets(pruning_dir, anchor)
    baseline_asr = None
    for ts, sz, a, _ in bf:
        if set(ts) == set(stage1_set):
            baseline_asr = a
            break
    if baseline_asr is None:
        return {tau: (list(stage1_set), None, None) for tau in taus}
    if removal_index is None:
        removal_index = index_remove_files(pruning_dir, anchor)

    current = set(stage1_set)
    history = [(frozenset(current), 0.0, baseline_asr)]
    while len(current) > 1:
        cands = []
        for name in current:
            # file tag encodes the set AFTER removal, so look up that frozenset
            after = frozenset(current - {name})
            key = (after, name)
            if key in removal_index:
                drop = baseline_asr - removal_index[key]
                cands.append((name, drop))
        if not cands:
            break
        cands.sort(key=lambda x: (x[1], x[0]))   # lowest_index tie-break
        chosen, drop = cands[0]
        new_set = current - {chosen}
        history.append((frozenset(new_set), drop, baseline_asr - drop))
        current = new_set
        if drop > max(taus):
            break

    out = {}
    for tau in taus:
        mech_set, mech_asr = stage1_set, baseline_asr
        for set_frozen, drop, new_asr in history:
            if drop <= tau:
                mech_set, mech_asr = list(set_frozen), new_asr
        out[tau] = (mech_set, mech_asr, len(history) - 1)
    return out


def _compact(name):
    """Inverse of parse_components for a single name."""
    if name == "embed":
        return "embed"
    if name.startswith("attn_"):
        return "a" + name.split("_", 1)[1]
    if name.startswith("mlp_"):
        return "m" + name.split("_", 1)[1]
    return name


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def collect_model(alias, anchor):
    """Return {key: (mech_set, val_asr, test_asr, test_mtime)} for one model."""
    d = os.path.join(RUNS_ROOT, alias, "completions", "pruning")
    if not os.path.isdir(d):
        return {}, None
    bf = find_baseline_full_sets(d, anchor)
    size_to_set = {}
    for ts, sz, a, _ in bf:
        size_to_set.setdefault(sz, (ts, a))
    sizes_sorted = sorted(size_to_set)
    alpha_to_set = {}
    if len(sizes_sorted) >= 3:
        alpha_to_set[0.90] = size_to_set[sizes_sorted[0]]
        alpha_to_set[0.95] = size_to_set[sizes_sorted[1]]
        alpha_to_set[0.99] = size_to_set[sizes_sorted[-1]]
    elif len(sizes_sorted) == 2:
        alpha_to_set[0.90] = size_to_set[sizes_sorted[0]]
        alpha_to_set[0.95] = size_to_set[sizes_sorted[1]]
    elif len(sizes_sorted) == 1:
        alpha_to_set[0.95] = size_to_set[sizes_sorted[0]]

    ri = index_remove_files(d, anchor)
    results = {}
    full_steering_files = glob.glob(os.path.join(d, "*_full_steering_test_evaluations.json"))
    full_steering_mtime = (os.path.getmtime(full_steering_files[0])
                           if full_steering_files else None)

    if 0.95 in alpha_to_set:
        s1, _ = alpha_to_set[0.95]
        mechs = replay_greedy(d, s1, [0.02, 0.05, 0.1], anchor, ri)
        for tau, (mech, val_asr, _) in mechs.items():
            tag = f"_exp1_a95_t{tau}_test_evaluations.json"
            te = glob.glob(os.path.join(d, f"*{tag}"))
            results[("exp1", 0.95, tau)] = (
                set(mech), val_asr,
                asr(te[0]) if te else None,
                os.path.getmtime(te[0]) if te else None,
            )

    for a_val in (0.90, 0.99):
        if a_val not in alpha_to_set:
            continue
        s1, _ = alpha_to_set[a_val]
        mechs = replay_greedy(d, s1, [0.05], anchor, ri)
        mech, val_asr, _ = mechs[0.05]
        a_tag = "0.9" if a_val == 0.90 else "0.99"
        tag = f"_exp2_a{a_tag}_t05_test_evaluations.json"
        te = glob.glob(os.path.join(d, f"*{tag}"))
        results[("exp2", a_val, 0.05)] = (
            set(mech), val_asr,
            asr(te[0]) if te else None,
            os.path.getmtime(te[0]) if te else None,
        )
    return results, full_steering_mtime


def main():
    import datetime as dt
    rows = []
    nesting_lines = []
    timing_lines = []
    for alias, anchor in MODELS:
        results, fs_mtime = collect_model(alias, anchor)
        for (exp, a, t), (mech, v, te, _) in sorted(results.items()):
            rows.append((alias, exp, a, t, len(mech), v, te))

        # nesting: tau-sweep at alpha=0.95 (expect: 0.02 ⊇ 0.05 ⊇ 0.10)
        e1 = {t: results[("exp1", 0.95, t)][0] for t in (0.02, 0.05, 0.1)
              if ("exp1", 0.95, t) in results}
        if {0.02, 0.05} <= set(e1):
            nesting_lines.append(_nesting_row(alias, "τ=0.02 ⊇ τ=0.05",
                                              e1[0.05], e1[0.02]))
        if {0.05, 0.1} <= set(e1):
            nesting_lines.append(_nesting_row(alias, "τ=0.05 ⊇ τ=0.10",
                                              e1[0.1], e1[0.05]))
        if {0.02, 0.1} <= set(e1):
            nesting_lines.append(_nesting_row(alias, "τ=0.02 ⊇ τ=0.10 (transitive)",
                                              e1[0.1], e1[0.02]))

        # alpha-sweep at tau=0.05 (expect: 0.90 ⊆ 0.95 ⊆ 0.99)
        a_mechs = {a: results[("exp2", a, 0.05)][0] for a in (0.90, 0.99)
                   if ("exp2", a, 0.05) in results}
        if ("exp1", 0.95, 0.05) in results:
            a_mechs[0.95] = results[("exp1", 0.95, 0.05)][0]
        elif ("exp2", 0.95, 0.05) in results:
            a_mechs[0.95] = results[("exp2", 0.95, 0.05)][0]
        if {0.90, 0.95} <= set(a_mechs):
            nesting_lines.append(_nesting_row(alias, "α=0.90 ⊆ α=0.95",
                                              a_mechs[0.90], a_mechs[0.95]))
        if {0.95, 0.99} <= set(a_mechs):
            nesting_lines.append(_nesting_row(alias, "α=0.95 ⊆ α=0.99",
                                              a_mechs[0.95], a_mechs[0.99]))

        # wall clock timeline of test files
        order = sorted(results.items(), key=lambda kv: kv[1][3] or float("inf"))
        prev = fs_mtime
        for (exp, a, t), (_, _, _, te) in order:
            if te is None:
                continue
            delta = (te - prev) / 60 if prev else None
            timing_lines.append((alias, exp, a, t,
                                 dt.datetime.fromtimestamp(te),
                                 f"{delta:.1f}m" if delta is not None else "—"))
            prev = te

    print("\n## Mechanism size + ASR per completed milestone")
    print("| model | exp | α | τ | mech_size | val ASR | test ASR |")
    print("|-|-|-|-|-|-|-|")
    for (alias, exp, a, t, ms, v, te) in rows:
        vs = f"{v:.3f}" if v is not None else "—"
        ts = f"{te:.3f}" if te is not None else "—"
        print(f"| {alias} | {exp} | {a} | {t} | {ms} | {vs} | {ts} |")

    print("\n## Nesting checks (strict subset of smaller ⊆ larger)")
    print("| model | relation | smaller size | larger size | intersection | strict subset? |")
    print("|-|-|-|-|-|-|")
    for line in nesting_lines:
        print(line)

    print("\n## Wall clock timeline of completed milestones (delta = time from previous)")
    print("| model | exp | α | τ | finished at | Δ from prev |")
    print("|-|-|-|-|-|-|")
    for (alias, exp, a, t, when, delta) in timing_lines:
        print(f"| {alias} | {exp} | {a} | {t} | {when:%m-%d %H:%M} | {delta} |")


def _nesting_row(alias, label, smaller, larger):
    inter = smaller & larger
    is_subset = smaller.issubset(larger)
    mark = "✓" if is_subset else "✗"
    return (f"| {alias} | {label} | {len(smaller)} | {len(larger)} | "
            f"{len(inter)} | {mark} |")


if __name__ == "__main__":
    main()
