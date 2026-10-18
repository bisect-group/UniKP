# UniKP emulator_bench

Multi-seed evaluation of UniKP with its original settings on the default EMULaToR splits.
Tasks: `kcat` and `km` (`data/processed/datasets/kinetic_dataset/<task>`). The original UniKP scripts are not changed.

## Model

- Protein: 1024-d mean-pooled ProtT5-XL-UniRef50 vector (fp32, frozen).
- Substrate: 1024-d SMILES Transformer vector (`vocab.pkl`, `trfm_12_23000.pkl`). The vector depends on the row position, see below.
- Predictor: sklearn `ExtraTreesRegressor()` with library defaults (100 trees, no depth limit), trained on the concatenated 2048-d vector. There is no validation use, no early stopping, and no tuning.
- Target: `log10_value`.

The wrapper sets only `n_jobs` and `random_state=seed`.

## Inputs and outputs

Splits are read from `<dataset_root>/<task>/<split_group>[/threshold_<t>]/{train,val,test}.parquet`.
Split groups: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits`, `substrate_splits`. Threshold folders are found on disk.

Everything the wrapper writes goes under `--work_dir` (default `outputs/baselines/UniKP`). It never writes to the split folders.

```
<work_dir>/embeddings/proteins/...                   shared cache, built once
<work_dir>/embeddings/smiles_trfm_v2_<fingerprint>/... one vector per (SMILES, chunk column)
<work_dir>/<task>/feature_matrices/<group>/<name>/   train/val/test .npz + metadata
<work_dir>/<task>/results/<group>/<name>/seed_<s>/   metrics, predictions, model.joblib
```

## SMILES Transformer rule

The original scripts encode the full, unsplit dataset in chunks of 100 rows. Its positional encoding slices along the batch axis, so a vector depends on the column of the row in the chunk. The wrapper copies this exactly:

- The full table is `<dataset_root>/<task>/<task>_kinetic_params_1d.parquet` (stored order, read only).
- Table row `r` gets the vector at column `r % 100`. One SMILES on several rows can have several vectors.
- A split row is linked to its table row by the key `ECNumber, uniprot_id, sequence, smiles, value, Organism, pH, Temperature, mutations, substrate, source, enzymeType`. The key is unique in both tables and every split row has a match. A row without a match uses column 0, and the count is printed.
- The cache holds one vector per (SMILES, column) pair. The cache folder name and the feature-matrix fingerprint contain a hash of the table SMILES column, so a changed table rebuilds them.

Check: on CPU the vectors for the first 250 rows of the kcat table equal the original `smiles_to_vec` output exactly (max abs diff 0.0).

Existing caches, matrices, and finished runs are skipped on rerun.

## Run

Seeds are 666 777 888. Run from the submodule root with the `mldb` environment.
The original kcat script drops multi-component SMILES (containing `.`) from the training data. The Km script does not. Use `--drop_dot_smiles_train` for kcat only.

```bash
CUDA_VISIBLE_DEVICES=1 python emulator_bench/launch_parallel_retrain_original.py --tasks kcat --drop_dot_smiles_train --num_workers 4 --model_n_jobs 8
CUDA_VISIBLE_DEVICES=1 python emulator_bench/launch_parallel_retrain_original.py --tasks km --num_workers 4 --model_n_jobs 8
python emulator_bench/aggregate_unikp_results.py
```

Use `--dataset_root` and `--work_dir` to change the paths. The same `--drop_dot_smiles_train` choice must be used in the cache and feature steps of one task (the launcher passes it on).
`--max_rows_per_split N` is for smoke tests only.
