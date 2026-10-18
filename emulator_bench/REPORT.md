# UniKP wrapper report

## Method
UniKP (Yu et al., Nat. Commun. 2023) predicts kcat, Km, and kcat/Km from an enzyme sequence and a substrate SMILES.
The protein is encoded by frozen ProtT5-XL-UniRef50 (last hidden layer, mean over residues, 1024-d).
Sequences longer than 1000 residues keep the first 500 and last 500. U, Z, O, B become X.
The substrate is encoded by a pre-trained SMILES Transformer (mean pool, max pool, first token of last layer, first token of penultimate layer; 4 x 256 = 1024-d).
The two vectors are concatenated (2048-d) and fed to an sklearn ExtraTreesRegressor.
The paper chose Extra Trees from 16 models (R2 0.65 for kcat) and used sklearn default parameters for all models (Methods, sklearn v1.1.1).
There is no neural training, no validation use, no early stopping.

## Original settings
| Item | Value | Source |
|---|---|---|
| Predictor | `ExtraTreesRegressor()` (sklearn defaults: 100 trees, max_depth None, min_samples_split 2, max_features 1.0, bootstrap False) | `UniKP_kcat_model.py:25`, `UniKP_Km.py:109`, `UniKP_kcat_Km.py:111`; paper Methods ("utilizing default parameters") |
| Random state | none (unseeded) | same lines |
| Epochs, lr, batch size, optimiser | not applicable | |
| Protein features | ProtT5-XL-UniRef50, fp32, one sequence at a time | `UniKP_kcat.py` `Seq_to_vec` |
| Substrate features | `trfm_12_23000.pkl`, `vocab.pkl`, 220 tokens, mean/max/first/penultimate-first | `UniKP_kcat.py` `smiles_to_vec`, `pretrain_trfm.py:_encode` |
| Target | log10(value) | `UniKP_kcat_model.py` |
| SMILES with `.` | dropped in kcat training script; not dropped in Km script | `UniKP_kcat_model.py` main loop; `UniKP_Km.py` |
| Eval protocol | 80/20 random split or 5-fold CV | `UniKP_Km.py:Kcat_predict`, `UniKP_kcat_Km.py` (replaced by EMULaToR splits) |

Paper and code agree. No difference to record.

## What the wrapper does now
`launch_parallel_retrain_original.py` runs three stages for the chosen tasks (kcat, km), split groups, and seeds (default 666 777 888):
1. `cache_embeddings.py`: embeds each unique sequence and each unique (SMILES, column) pair once (shared cache in `<work_dir>/embeddings`).
2. `build_split_features.py`: builds train/val/test 2048-d matrices per split folder.
3. `train_single_target_tvt.py`: fits `ExtraTreesRegressor(n_jobs, random_state=seed)` on train, reports train/val/test metrics.
Split groups: random_splits_grouped_sequence, random_splits_grouped_smiles, uniprot_time_splits, enzyme_sequence_splits/threshold_*, substrate_splits/threshold_* (found on disk: kcat 0.01 / 0.205, km 0.05 / 0.445).
All outputs go to `--work_dir` (default `outputs/baselines/UniKP`). Nothing is written to the split folders.
`aggregate_unikp_results.py` gives mean/std over seeds per task and split.

## Files
Removed: `emulator_bench/tune_optuna.py`, `emulator_bench/launch_parallel_retrain_from_optuna.py`, `Plan.md` (stale, described Optuna and old paths).
Rewritten: `common.py`, `cache_embeddings.py`, `build_split_features.py`, `train_single_target_tvt.py`, `launch_parallel_retrain_original.py`, `aggregate_unikp_results.py`, `README.md`, `commands.txt`.
Edited: `feature_pipeline.py`.
Removed from the code: Optuna paths and the hyperparameter-override flags (`--hparams_json`, `--n_estimators`, ...), `--random_state_mode`, the conformer/structure/group_shuffle split groups, the `UniKP_Kcat`/`UniKP_Km` sibling-folder layout, and the 5-seed default.
Original UniKP files outside `emulator_bench` are unchanged.

## Bugs fixed
- `transformers` 5.x has no `batch_encode_plus`; the ProtT5 call now uses `tokenizer(...)` (same ids, checked on a sample).
- Old defaults pointed to a missing path and to split groups that do not exist; discovery now matches the on-disk layout.
- Task layout: `--tasks` with `--dataset_root` replaces one `--base_dir` per task.
- Embeddings and features are no longer written next to the data; they use `--work_dir`.
- Targets are kept as float64 (were float32) so trees see the original precision.

## Deviations
- Mixed precision (bf16 autocast) and TF32 for the embedding step were removed, because the original ran fp32.
- ProtT5 runs in length-sorted batches with an attention mask instead of one sequence at a time. The mean pool uses only valid tokens, so the result is the same up to float rounding.
- Feature loading reads each unique cached vector once per split (speed only).
- `random_state=seed` is set so seeds differ in a repeatable way. The original is unseeded.
- `n_jobs` is a speed control and does not change the model.
- Multi-component SMILES: the filter is now applied to the train split only (`--drop_dot_smiles_train`, use for kcat), so every baseline is tested on the same test rows. Val and test rows are never dropped.

## Commands (all seeds x splits x tasks)
From `EMULaToR/baselines/UniKP`, env `/home/adhil/.conda/envs/mldb/bin/python`:
```
CUDA_VISIBLE_DEVICES=1 python emulator_bench/launch_parallel_retrain_original.py --tasks kcat --drop_dot_smiles_train --num_workers 4 --model_n_jobs 8
CUDA_VISIBLE_DEVICES=1 python emulator_bench/launch_parallel_retrain_original.py --tasks km --num_workers 4 --model_n_jobs 8
python emulator_bench/aggregate_unikp_results.py
```
Run order matters little. The first run builds the embedding cache (GPU, ProtT5 fp32).

## Changes after review

- `build_split_features.py`: every feature matrix now stores a fingerprint (source file path, size and mtime, column names, `--max_rows_per_split`, and the train-only `--drop_dot_smiles_train` flag). A matrix with a different or missing fingerprint is rebuilt, so changed flags never reuse stale features. Checked: a repeat run skipped all splits; the same run with `--drop_dot_smiles_train` rebuilt only train.
- `cache_embeddings.py` and `feature_pipeline.py`: embeddings are now written after each batch (`iter_protein_embeddings`, `iter_smiles_embeddings` yield per batch; files are atomic). A crash keeps all finished batches, and a restart embeds only the missing items.
- Not changed: `model.joblib` per run is large; there is no option to skip it. `--model_n_jobs` default stays single-thread. Summary output stays in the work dir.
- Smoke after the changes (mldb, GPU 1, 64 rows): cache, feature build twice (skip, then rebuild after a flag change), and the full launcher for kcat random_splits_grouped_sequence seed 666 finished. Output: `scratchpad/bench_smoke/UniKP/r2`.
- `common.py`, README, `commands.txt`: default `--work_dir` is now `<EMULaToR>/outputs/baselines/UniKP`, computed from the file location. Smoke: `DEFAULT_WORK_DIR` resolves to `/home/adhil/github/EMULaToR/outputs/baselines/UniKP`.

## Smoke test
Command: `launch_parallel_retrain_original.py --work_dir <scratch>/bench_smoke/UniKP --max_rows_per_split 64 --seeds 666 --split_groups random_splits_grouped_sequence enzyme_sequence_splits --model_n_jobs 4` on GPU 1, then the aggregator.
Result: cache, feature, train, and aggregate stages all finished for kcat and km (4 runs, metrics written). Run in env mldb (sklearn 1.8.0, transformers 5.19.0).
Split discovery was also checked for all five groups for both tasks.
No full training was run.

## Open issues
- Dot-SMILES: the Km script keeps them and the kcat script drops them from training. The README uses this per-task rule. Confirm it is wanted.
- Paper ablation rows for UniKP show std 0.00 for some axes. Extra Trees with 100 trees has low seed variance, so this is plausible, but it was not rerun.
- The sklearn version (1.8.0) is newer than the paper (1.1.1). Defaults are the same for the used parameters.
- Memory/time for full runs not measured: kcat train about 55k x 2048, km about 80k x 2048 per seed.

## SMILES Transformer features match the original

The original encodes the full dataset in chunks of 100 rows. Its positional encoding is added along the batch axis, so the vector of a row depends on its column in the chunk (`row index % 100`). The wrapper now does the same per row:

- Table row `r` of `<task>_kinetic_params_1d.parquet` gets column `r % 100`. Split rows find their table row by a 12-column key (unique in both tables, no split row missing). A missing row uses column 0 and the count is printed.
- Inputs are padded to 220 tokens, so a vector depends only on the SMILES and its column.
- Cache version 2: folder `smiles_trfm_v2_<fingerprint>` (hash of the table SMILES column). Feature-matrix fingerprints include it.
- Check: first 250 rows of the kcat table, CPU, original `smiles_to_vec` against wrapper vectors: max abs diff 0.0, 250 of 250 rows exact. Smoke run of both CLIs on 16 rows per split finished.
