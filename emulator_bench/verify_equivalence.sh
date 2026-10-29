#!/usr/bin/env bash
# Checks that the UniKP wrapper matches the original code. Usage: verify_equivalence.sh [--gpu]
# Expected env: mldb (torch, transformers, sentencepiece, sklearn, pytest). The script does not create envs.
# CPU mode: tiny random T5 and the committed fixture; any skipped test fails the run.
# --gpu mode: also checks the real SMILES cache and the real ProtT5 one-at-a-time path on cuda:0. It needs the EMULaToR data,
# the built SMILES cache, the CataPro ProtT5 cache and the Rostlab/prot_t5_xl_uniref50 weights in the local Hugging Face cache.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: mldb"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
pytest_args=(-q -rs -p no:cacheprovider --color=no -o addopts="")
out=$(CUDA_VISIBLE_DEVICES="" python -m pytest tests "${pytest_args[@]}" -m "not gpu" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
if [[ ${1:-} == --gpu ]]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m pytest tests "${pytest_args[@]}" -s -m gpu
fi
