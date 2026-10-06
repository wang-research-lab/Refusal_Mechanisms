#!/usr/bin/env python3
"""Crawl logs/ for Inspect .eval archives and emit human-readable JSON.

Outputs under pipeline/parsed_logs/:
  <family>/<role>/<bench>.json   per-eval: metadata, metrics, per-sample rows
  _summary.json                  one row per eval — flat list, easy to load
  _summary.md                    markdown table for skimming

Per-sample text fields (input/completion/target) are truncated to 500 chars
to keep files manageable. For full sample text use `inspect log dump <path>`.

Layout assumptions (matches the 2026-05-19 reorganization):
  logs/models/<family>/<role>/<bench>/*.eval        (canonical)
  logs/<bench>_<family>-<role>/*.eval               (flat, in-flight / legacy)

Usage:
    python -m pipeline.parse_eval_logs                       # parse everything
    python -m pipeline.parse_eval_logs --bench mmlu          # one bench only
    python -m pipeline.parse_eval_logs --family glm          # one family
    python -m pipeline.parse_eval_logs --no-samples          # metadata only
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

from inspect_ai.log import read_eval_log

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "pipeline" / "parsed_logs"
TRUNC = 500


def truncate(s, n: int = TRUNC):
    if s is None:
        return None
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"...[truncated, total {len(s)} chars]"


def extract_metrics(log) -> dict:
    """Aggregate metrics — {scorer_name: {metric_name: value}}.

    For evals that were killed before the metrics block was finalised
    (status="started"), we fall back to computing a per-scorer accuracy
    over whatever samples did complete. The fallback values land under
    keys like "accuracy_partial" so the consumer can tell them apart from
    the canonical metrics block.
    """
    out: dict = {}
    if log.results and log.results.scores:
        for sc in log.results.scores:
            d: dict = {}
            for mname, mobj in (sc.metrics or {}).items():
                try:
                    d[mname] = mobj.value
                except AttributeError:
                    d[mname] = mobj
            out[sc.name] = d

    # Fallback: if no metrics (killed mid-run), tally per-scorer correct/total
    # over the samples we did get.
    if not out and log.samples:
        per_scorer_correct: dict = {}
        per_scorer_total: dict = {}
        for s in log.samples:
            if s.error:
                continue
            for scorer_name, sc in (s.scores or {}).items():
                try:
                    val = sc.value
                except AttributeError:
                    val = sc
                per_scorer_total[scorer_name] = per_scorer_total.get(scorer_name, 0) + 1
                if val in ("C", "c", True, 1):
                    per_scorer_correct[scorer_name] = per_scorer_correct.get(scorer_name, 0) + 1
                elif isinstance(val, (int, float)) and val not in (0, 1):
                    # numeric scorer (e.g. RedCode 0-10): track sum for mean
                    per_scorer_correct[scorer_name] = per_scorer_correct.get(scorer_name, 0.0) + val
        for scorer_name, total in per_scorer_total.items():
            num = per_scorer_correct.get(scorer_name, 0)
            partial = num / total if total else None
            out[scorer_name] = {
                "accuracy_partial": partial,
                "scored_n_partial": total,
            }
    return out


def primary_score(sample):
    """Pick the first scorer's value, if any."""
    sc = getattr(sample, "scores", None) or {}
    if not sc:
        return None
    first = next(iter(sc.values()))
    try:
        return first.value
    except AttributeError:
        return first


def parse_eval(path: pathlib.Path) -> dict:
    log = read_eval_log(str(path))
    samples: list[dict] = []
    for s in (log.samples or []):
        comp = s.output.completion if s.output else None
        samples.append({
            "id": s.id,
            "target": truncate(s.target),
            "score": primary_score(s),
            "completion": truncate(comp),
            "error": str(s.error) if s.error else None,
        })
    return {
        "task": log.eval.task,
        "model": log.eval.model,
        "status": log.status,
        "created": str(log.eval.created),
        "total_samples": (log.eval.dataset.samples if log.eval.dataset else None),
        "completed_samples": len(samples),
        "errored_samples": sum(1 for r in samples if r["error"]),
        "metrics": extract_metrics(log),
        "samples": samples,
    }


# Benches that serve_and_eval.sh can run. The flat log dir is "<bench>_<alias>";
# we identify the bench by longest-matching prefix (so the redcode_gen underscore
# wins over a shorter match) and treat whatever follows as the served alias.
_KNOWN_BENCHES = (
    "agentharm", "asb", "fortress", "agentdojo", "codeipi", "bfcl",
    "mmlu", "gsm8k", "labbench", "wmdp", "redcode_gen",
)
_KNOWN_BENCHES_BY_LEN = sorted(_KNOWN_BENCHES, key=len, reverse=True)


def _split_flat_dir(name: str) -> tuple[str | None, str | None, str | None]:
    """logs/<bench>_<family>-<role>  →  (family, role, bench).

    bench comes from the known set (family-agnostic, so gemma/mistral/phi/qwen
    all work); the remaining alias splits on its last hyphen into family/role,
    e.g. "gemma-base" → ("gemma", "base"), "Mistral-Small-base" → ("Mistral-Small", "base").
    """
    for bench in _KNOWN_BENCHES_BY_LEN:
        prefix = bench + "_"
        if name.startswith(prefix):
            alias = name[len(prefix):]
            family, sep, role = alias.rpartition("-")
            if not sep:  # no role hyphen — treat the whole alias as the family
                family, role = alias, ""
            return (family or None), (role or None), bench
    return None, None, None


def infer_family_role_bench(p: pathlib.Path) -> tuple[str | None, str | None, str | None]:
    """Given an .eval path, infer (family, role, bench)."""
    parts = p.parts
    if "logs" not in parts:
        return None, None, None
    rel = parts[parts.index("logs") + 1:]
    if not rel:
        return None, None, None
    if rel[0] == "models" and len(rel) >= 4:
        return rel[1], rel[2], rel[3]
    return _split_flat_dir(rel[0])


def _fmt_metric(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return str(v)


# Metric names that survive into the markdown summary. The per-eval JSON keeps
# everything; this filter just makes the at-a-glance table readable.
_HEADLINE_KEEP = {
    "accuracy", "precision", "coverage",
    "avg_score", "avg_refusals", "avg_full_score", "avg_score_non_refusals",
    "injection_resistance_rate", "task_completion_rate", "detection_rate",
}


def _is_headline_metric(metric_name: str) -> bool:
    """Drop noisy subgroup/stderr metrics in the markdown view."""
    n = metric_name.lower()
    if "stderr" in n:
        return False
    if "subdomain" in n or "groupkey" in n:
        # Keep only the "overall" rollup of subgroup metrics.
        return "overall" in n
    if metric_name in _HEADLINE_KEEP:
        return True
    # Fall-through: keep short, leaf-like metric names.
    return len(metric_name) <= 24


def write_markdown(summary: list[dict], path: pathlib.Path) -> None:
    lines = [
        "# Eval results summary",
        "",
        "Headline metrics only — see per-eval JSON for the full breakdown.",
        "",
        "| family | role | bench | status | done/total | errored | metrics |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        flat = []
        for scorer, mdict in (r["metrics"] or {}).items():
            for k, v in mdict.items():
                if not _is_headline_metric(k):
                    continue
                flat.append(f"{k}={_fmt_metric(v)}")
        ms = ", ".join(flat) if flat else "—"
        lines.append(
            f"| {r['family']} | {r['role']} | {r['bench']} | {r['status']} | "
            f"{r['completed']}/{r['total']} | {r['errored']} | {ms} |"
        )
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench")
    ap.add_argument("--family")
    ap.add_argument("--role")
    ap.add_argument("--no-samples", action="store_true",
                    help="omit the samples[] array from per-eval JSON")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    eval_paths = sorted((ROOT / "logs").rglob("*.eval"))
    print(f"Found {len(eval_paths)} .eval files")

    summary: list[dict] = []
    written_names: set[pathlib.Path] = set()
    for p in eval_paths:
        family, role, bench = infer_family_role_bench(p)
        if not (family and role and bench):
            print(f"  skip (cannot infer layout): {p}")
            continue
        if args.bench and bench != args.bench:
            continue
        if args.family and family != args.family:
            continue
        if args.role and role != args.role:
            continue

        try:
            data = parse_eval(p)
        except Exception as e:
            print(f"  ERROR {p}: {e}")
            traceback.print_exc()
            continue

        out_dir = OUT / family / role
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{bench}.json"
        out_path = out_dir / out_name
        # Disambiguate when one bench has multiple .eval files (e.g. fortress
        # adversarial vs benign, or labbench's per-subset files).
        if out_path in written_names:
            tail = p.stem.split("_")[-1][:10]
            out_path = out_dir / f"{bench}__{tail}.json"
        written_names.add(out_path)

        payload = dict(data)
        if args.no_samples:
            payload.pop("samples", None)
        out_path.write_text(json.dumps(payload, indent=2, default=str))

        summary.append({
            "family": family, "role": role, "bench": bench,
            "task": data["task"], "status": data["status"],
            "total": data["total_samples"],
            "completed": data["completed_samples"],
            "errored": data["errored_samples"],
            "metrics": data["metrics"],
            "source": str(p.relative_to(ROOT)),
            "parsed": str(out_path.relative_to(ROOT)),
        })
        print(f"  {family:8s} {role:6s} {bench:12s} {data['status']:10s} "
              f"{data['completed_samples']:>5d}/{data['total_samples']}  err={data['errored_samples']}")

    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_markdown(summary, OUT / "_summary.md")
    print(f"\nWrote {len(summary)} parsed evals to {OUT}")
    print(f"  Summary table: {OUT / '_summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
