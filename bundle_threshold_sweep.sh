#!/usr/bin/env bash
# Bundle the threshold-sweep / seed-stability outputs (rebuttal side-claim) into a
# self-contained tree at pipeline/runs_threshold_sweep/, so the side-claim
# artifacts can be reviewed separately from the paper's main-claim files in
# pipeline/runs/. Originals are COPIED, never moved — the content-addressed cache
# and the paper's tracked artifacts stay untouched in pipeline/runs/.
#
# Run AFTER all sweeps + `python -m pipeline.threshold_seed_sweep --report` complete.
#
# How sweep-added files are identified:
#   - Combined report: pipeline/runs/rebuttal_threshold_seed_results.{json,md}
#   - Per-model sidecar (final exp1+exp2+exp3 records):
#       pipeline/runs/<model>/threshold_seed_sidecar.json
#   - Everything else lives under completions/pruning/ and is identified as
#     git-untracked. (Your paper's pruning files are tracked, so untracked = sweep.)
#     Those split into:
#       * test_evals/  — *_exp[123]_*_test_* and *_full_steering_test_* eval/completion
#                        JSONs (the final ASR numbers per tau / alpha / seed).
#       * trajectory/  — the per-removal evals along the Stage-2 trajectory (these
#                        share the paper's CompTest_…ablate_remove_X naming).
#
# Env:
#   DRY_RUN=1   — preview, don't actually copy.
#   SRC=…       — override the source tree (default pipeline/runs).
#   DST=…       — override the destination tree (default pipeline/runs_threshold_sweep).

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

SRC="${SRC:-pipeline/runs}"
DST="${DST:-pipeline/runs_threshold_sweep}"
DRY="${DRY_RUN:-0}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "error: not in a git repo — script relies on 'git ls-files --others' to identify sweep-added files." >&2
    exit 1
fi

copy() {  # copy src -> dst, preserving mtime; respect DRY
    local s="$1" d="$2"
    if [ "$DRY" = "1" ]; then
        echo "  [dry] $s -> $d"
    else
        mkdir -p "$(dirname "$d")"
        cp -p "$s" "$d"
    fi
}

mkdir -p "$DST"

# 1) Combined report at the runs root.
for ext in json md; do
    f="$SRC/rebuttal_threshold_seed_results.$ext"
    [ -f "$f" ] && copy "$f" "$DST/rebuttal_threshold_seed_results.$ext"
done

# 2) Per-model: sidecar + sweep-added files under completions/pruning/, split into
#    test_evals/ and trajectory/.
n_models=0
n_test=0
n_traj=0
for mdir in "$SRC"/*/; do
    [ -d "$mdir" ] || continue
    model=$(basename "$mdir")
    sidecar="$mdir/threshold_seed_sidecar.json"
    [ -f "$sidecar" ] || continue          # model not swept; skip
    copy "$sidecar" "$DST/$model/threshold_seed_sidecar.json"

    # All sweep-added (= git-untracked) files under this model's pruning dir.
    # Split by filename pattern.
    while IFS= read -r f; do
        [ -n "$f" ] && [ -f "$f" ] || continue
        base=$(basename "$f")
        case "$base" in
            *_exp1_*|*_exp2_*|*_exp3_*|*_full_steering_test_*)
                copy "$f" "$DST/$model/test_evals/$base"; n_test=$((n_test+1)) ;;
            *)
                copy "$f" "$DST/$model/trajectory/$base"; n_traj=$((n_traj+1)) ;;
        esac
    done < <(git ls-files --others --exclude-standard "$mdir/completions/pruning/" 2>/dev/null || true)

    n_models=$((n_models+1))
done

# 3) README so reviewers know what they're looking at.
README="$DST/README.md"
if [ "$DRY" != "1" ]; then
cat > "$README" <<'MD'
# Threshold-sweep bundle (rebuttal side-claim)

Mirrors the threshold-sweep + seed-stability outputs from `pipeline/runs/` into a
self-contained tree so the side-claim results can be reviewed without mixing
with the paper's main-claim artifacts. All files here were **copied** from
`pipeline/runs/`; the originals stay in place so the content-addressed cache
and the paper's tracked files remain untouched.

## Layout
- `rebuttal_threshold_seed_results.json` / `.md` — combined summary across all
  models, plus the two markdown tables (Table 1 threshold sensitivity,
  Table 2 seed stability).
- `<model>/threshold_seed_sidecar.json` — that model's exp1 (tau-sweep at
  alpha=0.95), exp2 (alpha-sweep at tau=0.05), exp3 (seed stability) records,
  including validation_flags.
- `<model>/test_evals/` — final ASR evaluations on the 512-prompt test set for
  each mechanism the sweep identified, plus the full-steering reference ASR.
  Tag patterns:
    *_exp1_a95_t{0.02,0.05,0.10}_test_*    tau-sweep final mechanism on test
    *_exp2_a{0.90,0.99}_t05_test_*         alpha-sweep final mechanism on test
    *_exp3_s{0,1,2}_test_*                 per-seed final mechanism on test
    *_full_steering_test_*                  full-component (2*L+1) reference
- `<model>/trajectory/` — the per-removal completions + LlamaGuard3 evals along
  the Stage-2 elimination trajectory (content-addressed CompTest tags). These
  are the intermediate scratch the sweep added; the paper's pre-existing
  trajectory files (which the sweep reused via cache) are NOT mirrored here.

## How files are identified
The paper's `pipeline/runs/` artifacts are git-tracked, so the sweep-added files
are exactly the union of:
- `threshold_seed_sidecar.json` and `rebuttal_threshold_seed_results.{json,md}`
  (named outputs), and
- everything `git ls-files --others --exclude-standard pipeline/runs/<model>/completions/pruning/`
  returns (untracked under `pruning/`).

This bundle is regenerated by `bundle_threshold_sweep.sh` at repo root.
MD
fi

if [ "$DRY" = "1" ]; then
    echo "[dry] $n_models swept models | $n_test test eval files | $n_traj trajectory files"
    echo "[dry] would write to $DST/"
else
    echo "Wrote bundle to $DST/  ($n_models models, $n_test test-eval files, $n_traj trajectory files)"
    du -sh "$DST" 2>/dev/null
    ls "$DST"
fi
