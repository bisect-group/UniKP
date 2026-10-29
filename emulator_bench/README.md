# UniKP emulator_bench

Multi-seed evaluation of UniKP with its original settings on the default EMULaToR splits.
Tasks: `kcat` and `km` (`data/processed/datasets/kinetic_dataset/<task>`). The original UniKP scripts are not changed.
Commands: `commands.txt`. Full description, numbers and open issues: `REPORT.md`.

## Model

- Protein: 1024-d mean-pooled ProtT5-XL-UniRef50 vector (fp32, frozen).
- Substrate: 1024-d SMILES Transformer vector (`vocab.pkl`, `trfm_12_23000.pkl`). The vector depends on the row position, see below.
- Predictor: sklearn `ExtraTreesRegressor()` with library defaults (100 trees, no depth limit), trained on the concatenated 2048-d vector (SMILES first, protein second). There is no validation use, no early stopping, and no tuning.
- Target: `log10_value`.

The wrapper sets only `n_jobs` and `random_state=seed`.

## Inputs and outputs

Splits are read from `<dataset_root>/<task>/<split_group>[/threshold_<t>]/{train,val,test}.parquet`.
Split groups: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits`, `enzyme_structure_splits`, `substrate_splits`, `conformer_cosine_splits`. Threshold folders are found on disk. A group without threshold folders uses `all` as `<name>`. This gives 7 split folders per task, so 21 runs per task and 42 in total.

Everything the wrapper writes goes under `--work_dir` (default `outputs/baselines/UniKP`). It never writes to the split folders.
The only other write is the ProtT5 cache, see below.

```
<protein_cache_dir>/<sha256>.npy                      ProtT5 vectors, shared with CataPro
<work_dir>/embeddings/smiles_trfm_v5_<fingerprint>/... one vector per (SMILES, chunk column)
<work_dir>/<task>/feature_matrices/<group>/<name>/    train/val/test .npz + metadata
<work_dir>/<task>/<group>/<name>/seed_<s>/            metrics, predictions, model.joblib, run_summary.json
```

## Protein vectors

`--protein_cache_dir` defaults to `outputs/baselines/CataPro/embeddings/prot_t5_v2`. The wrapper reads it in place with the CataPro key: `sha256("v2|<model id>|max_len=1000|float32|<text>")`, where `<text>` is the processed sequence (first 500 plus last 500 residues if longer than 1000, U/Z/O/B to X, residues joined by spaces).
A missing vector is computed with the same batched path (batch 8, length-sorted, padded to the longest, fp32 mean over residues) and written to the same folder.
The original code embeds one sequence at a time and averages in float64. Results differ at float rounding level, see `REPORT.md`.

## SMILES Transformer rule

The original scripts encode the full, unsplit dataset in chunks of 100 rows. Its positional encoding slices along the batch axis, so a vector depends on the column of the row in the chunk. The wrapper copies this exactly:

- The full table is `<dataset_root>/<task>/<task>_kinetic_params_1d.parquet` (stored order, read only).
- Table row `r` gets the vector at column `r % 100`. One SMILES on several rows can have several vectors.
- A split row is linked to its table row by the key `ECNumber, uniprot_id, sequence, smiles, value, Organism, mutations, substrate, source, enzymeType`. The key is unique in both tables. A split row without a match stops the run with an error.
- The cache folder name and the feature-matrix fingerprint contain a hash of the table SMILES column, so a changed table rebuilds them.

The vectors are built on the CPU (`--smiles_device cpu`, the default), like the original, and equal the original `smiles_to_vec` output exactly (`tests/`). Building both tasks takes about 3.5 minutes on 48 threads.

Existing caches, matrices, and finished runs are skipped on rerun.

## Run

Seeds are 666 777 888. Run from the submodule root with the `mldb` environment.
The original kcat script drops multi-component SMILES (containing `.`) from the training data. The Km script does not. Use `--drop_dot_smiles_train` for kcat only. Val and test rows are never dropped.

```bash
PY=python
$PY emulator_bench/launch_parallel_retrain_original.py --dry_run
$PY emulator_bench/launch_parallel_retrain_original.py --tasks kcat --drop_dot_smiles_train --jobs 4 --model_n_jobs 12
$PY emulator_bench/launch_parallel_retrain_original.py --tasks km --jobs 4 --model_n_jobs 12
$PY emulator_bench/aggregate_unikp_results.py
```

`--jobs N` runs N trainings at once. Extra Trees trains on CPU, so the launcher has no `--gpus`, `--runs_per_gpu` or `--amp`. Only missing ProtT5 vectors use a GPU (`--cache_device`); the SMILES step runs on the CPU (`--smiles_device`).
Set `--model_n_jobs` so that `jobs x model_n_jobs` fits the CPU threads. `--jobs 4 --model_n_jobs 12` fits 48 threads. See `REPORT.md` for timings.
Each run writes `model.joblib` (about 0.6 GB for kcat) and `run_summary.json` with fit, dump and predict times and peak memory.

Use `--dataset_root`, `--work_dir` and `--protein_cache_dir` to change the paths. The same `--drop_dot_smiles_train` choice must be used in the cache and feature steps of one task (the launcher passes it on).
`--max_rows_per_split N` is for smoke tests only.

## Check against the original code

```bash
bash emulator_bench/verify_equivalence.sh          # CPU, tiny random T5, committed fixture in tests/fixtures
bash emulator_bench/verify_equivalence.sh --gpu    # also: real SMILES cache vs original, real ProtT5 one-at-a-time path on cuda:0
```

The tests call the original `smiles_to_vec`, `Seq_to_vec` and `ExtraTreesRegressor()` and compare them with the wrapper.
