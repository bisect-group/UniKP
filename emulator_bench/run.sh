#!/usr/bin/env bash
# Reproduce UniKP: ProtT5 and SMILES embeddings, feature matrices, Extra Trees runs (kcat, km), seed summary.
# Usage: emulator_bench/run.sh [--dry-run] [cache|train|aggregate|all]
# Env: PY, GPUS (one id, only for missing ProtT5 vectors), SEEDS, JOBS, MODEL_N_JOBS, DATA_ROOT, OUT_ROOT, DRY_RUN=1.
# cache = embeddings + feature matrices. train also evaluates; finished work is skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=${PY:-python}
read -ra SEEDS <<<"${SEEDS:-666 777 888}"
JOBS=${JOBS:-3}
MODEL_N_JOBS=${MODEL_N_JOBS:-12}
DATA=() OUT=()
[[ -z ${DATA_ROOT:-} ]] || DATA=(--dataset_root "$DATA_ROOT")
[[ -z ${OUT_ROOT:-} ]] || OUT=(--work_dir "$OUT_ROOT")
STEP=all
for a in "$@"; do
  case $a in
    --dry-run) DRY_RUN=1 ;;
    cache | train | aggregate | all) STEP=$a ;;
    *) echo "usage: $0 [--dry-run] [cache|train|aggregate|all]" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

export CUDA_VISIBLE_DEVICES=${GPUS:-0}

# The original kcat script drops multi-component SMILES from train; the Km script keeps them.
for task in kcat km; do
  flags=()
  [[ $task != kcat ]] || flags=(--drop_dot_smiles_train)
  if want cache; then
    run "$PY" emulator_bench/cache_embeddings.py --tasks "$task" "${flags[@]}" "${DATA[@]}" "${OUT[@]}"
    run "$PY" emulator_bench/build_split_features.py --tasks "$task" "${flags[@]}" "${DATA[@]}" "${OUT[@]}"
  fi
  if want train; then
    run "$PY" emulator_bench/launch_parallel_retrain_original.py --tasks "$task" "${flags[@]}" --skip_cache --skip_feature_build \
      --seeds "${SEEDS[@]}" --jobs "$JOBS" --model_n_jobs "$MODEL_N_JOBS" "${DATA[@]}" "${OUT[@]}"
  fi
done
if want aggregate; then
  run "$PY" emulator_bench/aggregate_unikp_results.py "${OUT[@]}"
fi
