"""Probe threshold_seed_sweep progress without touching running jobs.

Reads test_asr evaluation files (uniquely tagged per fresh run) to count
completed experiment milestones definitively. Sidecars only land at the end
of all three experiments, so this is the only way to see partial progress.

Each fresh trajectory writes a distinct *_test_evaluations.json file when its
test ASR completes:
  full_steering_test                      — script start
  exp1_a95_t{0.02,0.05,0.1}_test         — Exp 1 tau-sweep stopping points
  exp2_a{0.9,0.99}_t05_test              — Exp 2 fresh alpha runs
                                          (alpha=0.95 borrows Exp 1 tau=0.05)
  exp3_s{0,1,2}_test                     — Exp 3 seed runs

Usage:
  python -m pipeline.probe_sweep
  python -m pipeline.probe_sweep --since 2026-05-28T09:45   # only files after restart
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re

MODELS = [
    ("gemma-2-2b-it", 13),
    ("Phi-4-mini-instruct", 14),
    ("Qwen3-4B", 19),
    ("Mistral-Small-3.2-24B-Instruct-2506", 17),
]

# Required test-file markers for a complete run.
MILESTONES = [
    ("full_steering",     "_full_steering_test"),
    ("exp1 tau=0.02",     "_exp1_a95_t0.02_test"),
    ("exp1 tau=0.05",     "_exp1_a95_t0.05_test"),
    ("exp1 tau=0.10",     "_exp1_a95_t0.1_test"),
    ("exp2 alpha=0.90",   "_exp2_a0.9_t05_test"),
    ("exp2 alpha=0.99",   "_exp2_a0.99_t05_test"),
    ("exp3 seed=0",       "_exp3_s0_test"),
    ("exp3 seed=1",       "_exp3_s1_test"),
    ("exp3 seed=2",       "_exp3_s2_test"),
]

RUNS_ROOT = "pipeline/runs"


def _mtime(path):
    return dt.datetime.fromtimestamp(os.path.getmtime(path))


def probe_model(alias, anchor, since=None):
    model_dir = os.path.join(RUNS_ROOT, alias)
    d = os.path.join(model_dir, "completions", "pruning")
    if not os.path.isdir(d):
        return {"alias": alias, "missing_dir": True}

    test_files = glob.glob(os.path.join(d, "*_test_evaluations.json"))
    by_marker = {}
    for path in test_files:
        name = os.path.basename(path)
        for label, marker in MILESTONES:
            if marker + "_evaluations.json" in name:
                by_marker[label] = _mtime(path)
                break

    # Trajectory file activity (rate signal for the in-flight fresh run).
    traj = glob.glob(os.path.join(d, "harmful_CompTest_*_completions.json"))
    traj_evals = glob.glob(os.path.join(d, "harmful_CompTest_*_evaluations.json"))
    all_files = traj + traj_evals
    if since:
        all_files = [f for f in all_files if _mtime(f) >= since]
    fresh_count = len(all_files)
    latest = max((_mtime(f) for f in all_files), default=None)

    # Sidecar shards (interim + final). Patched runs write _status field.
    shards = []
    for sc in sorted(glob.glob(os.path.join(model_dir, "threshold_seed_sidecar*.json"))):
        try:
            block = json.load(open(sc))
        except (json.JSONDecodeError, OSError):
            continue
        shards.append({
            "path": os.path.basename(sc),
            "status": block.get("_status"),
            "mtime": _mtime(sc),
            "exp1_keys": [k for k in block.get("exp1", {}) if k.startswith("tau_")],
            "exp2_keys": [k for k in block.get("exp2", {}) if k.startswith("alpha_")],
            "exp3_keys": [k for k in block.get("exp3", {}) if k.startswith("seed_")],
        })

    return {
        "alias": alias,
        "anchor": anchor,
        "milestones": by_marker,
        "fresh_count_since": fresh_count,
        "latest_activity": latest,
        "shards": shards,
    }


def format_report(probes, since=None):
    lines = []
    now = dt.datetime.now()
    lines.append(f"# Sweep probe at {now:%Y-%m-%d %H:%M:%S}")
    if since:
        lines.append(f"  (rate signal counts files written since {since:%Y-%m-%d %H:%M})")
    lines.append("")

    for p in probes:
        if p.get("missing_dir"):
            lines.append(f"## {p['alias']}  (no run dir)")
            continue
        lines.append(f"## {p['alias']}  (L{p['anchor']})")
        done = [label for label, _ in MILESTONES if label in p["milestones"]]
        todo = [label for label, _ in MILESTONES if label not in p["milestones"]]
        lines.append(f"  done ({len(done)}/{len(MILESTONES)}): {', '.join(done) or '(none)'}")
        lines.append(f"  todo ({len(todo)}): {', '.join(todo) or '(none)'}")
        # milestone timeline
        for label, _ in MILESTONES:
            t = p["milestones"].get(label)
            if t:
                lines.append(f"    {t:%Y-%m-%d %H:%M}  {label}")
        # latest activity / rate
        if p["latest_activity"]:
            stale = now - p["latest_activity"]
            stale_str = f"{int(stale.total_seconds() // 60)}m ago"
            lines.append(f"  latest fresh file: {p['latest_activity']:%H:%M} ({stale_str})")
            lines.append(f"  fresh files since marker: {p['fresh_count_since']}")
        # sidecar shards
        for sh in p.get("shards", []):
            keys = []
            if sh["exp1_keys"]: keys.append(f"exp1={'/'.join(sh['exp1_keys'])}")
            if sh["exp2_keys"]: keys.append(f"exp2={'/'.join(sh['exp2_keys'])}")
            if sh["exp3_keys"]: keys.append(f"exp3={'/'.join(sh['exp3_keys'])}")
            keys_str = ", ".join(keys) or "(empty)"
            lines.append(f"  shard {sh['path']}  status={sh['status'] or '?'}  {keys_str}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None,
                    help="ISO timestamp; only count fresh files after this (e.g. 2026-05-28T09:45)")
    args = ap.parse_args()

    since = dt.datetime.fromisoformat(args.since) if args.since else None
    probes = [probe_model(a, l, since=since) for a, l in MODELS]
    print(format_report(probes, since=since))


if __name__ == "__main__":
    main()
