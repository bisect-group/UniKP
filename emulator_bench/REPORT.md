# UniKP wrapper report

Machine for all numbers: 24 cores / 48 threads, 123 GB RAM, one RTX PRO 4500 (32 GB). Env `mldb` (sklearn 1.8.0, transformers 5.19.0, torch 2.12).

## 1. Method
UniKP (Yu et al., Nat. Commun. 2023) predicts kcat, Km, and kcat/Km from an enzyme sequence and a substrate SMILES.
The protein is encoded by frozen ProtT5-XL-UniRef50 (last hidden layer, mean over residues, 1024-d).
Sequences longer than 1000 residues keep the first 500 and last 500. U, Z, O, B become X.
The substrate is encoded by a pre-trained SMILES Transformer (mean pool, max pool, first token of last layer, first token of penultimate layer; 4 x 256 = 1024-d).
The two vectors are concatenated (SMILES first, protein second; 2048-d) and fed to an sklearn ExtraTreesRegressor.
The paper chose Extra Trees from 16 models (R2 0.65 for kcat) and used sklearn default parameters for all models (Methods, sklearn v1.1.1).
There is no neural training, no validation use, no early stopping.

## 2. Original settings
| Item | Value | Source |
|---|---|---|
| Predictor | `ExtraTreesRegressor()` (sklearn defaults: 100 trees, max_depth None, min_samples_split 2, max_features 1.0, bootstrap False) | `UniKP_kcat.py` `Kcat_predict`, `UniKP_kcat_model.py`, `UniKP_Km.py` `Kcat_predict`; paper Methods ("utilizing default parameters") |
| Random state | none (unseeded) | same lines |
| Epochs, lr, batch size, optimiser | not applicable | |
| Protein features | ProtT5-XL-UniRef50, fp32, one sequence at a time, float64 mean over residues | `UniKP_kcat.py` `Seq_to_vec` |
| Substrate features | `trfm_12_23000.pkl`, `vocab.pkl`, 220 tokens, mean/max/first/penultimate-first, chunks of 100 rows | `UniKP_kcat.py` `smiles_to_vec`, `pretrain_trfm.py` `_encode`, `encode` |
| Target | log10(value) | `UniKP_kcat.py` main |
| SMILES with `.` | dropped from the training data in the kcat script (`'.' not in Smiles[i]`); not dropped in the Km script | `UniKP_kcat.py` main; `UniKP_Km.py` main |
| Eval protocol | 90/10 random split x 10 (kcat), 80/20 random split (Km) | `Kcat_predict` (replaced by EMULaToR splits) |

Paper and code agree. No difference to record.

## 3. What the wrapper does
`launch_parallel_retrain_original.py` runs three stages per task (kcat, km), split groups and seeds (666, 777, 888):
1. `cache_embeddings.py`: reads ProtT5 vectors from the CataPro cache (`outputs/baselines/CataPro/embeddings/prot_t5_v2`, same key function) and computes any missing vector with the same batched path. It embeds each unique (SMILES, column) pair once on the CPU (`--smiles_device cpu`, the default), as the original does.
2. `build_split_features.py`: builds the train/val/test 2048-d matrices of all split folders of one task in one process. Each unique vector is loaded once per task.
3. `train_single_target_tvt.py`: fits `ExtraTreesRegressor(n_jobs, random_state=seed)` on train, reports train/val/test metrics, saves predictions and `model.joblib`. `run_summary.json` holds fit, dump and predict times and peak memory.

The SMILES Transformer vector depends on the row position. The original encodes the full table in chunks of 100 rows, and the positional encoding is added along the batch axis. Table row `r` therefore gets the vector of chunk column `r % 100`. A split row finds its table row by the 10-column key `ECNumber, uniprot_id, sequence, smiles, value, Organism, mutations, substrate, source, enzymeType` (unique in both tables). A split row without a match stops the run.

Split groups (found on disk): random_splits_grouped_sequence, random_splits_grouped_smiles, uniprot_time_splits, enzyme_sequence_splits, enzyme_structure_splits, substrate_splits, conformer_cosine_splits. Each task has 7 split folders (kcat thresholds 0.01 / 0.625 / 0.505 / 0.996, km 0.05 / 0.69 / 0.52 / 0.997), so 7 x 3 seeds x 2 tasks = 42 runs (`--dry_run` lists them).
Multi-component SMILES: `--drop_dot_smiles_train` for kcat only, as in the original scripts. Val and test keep all rows. Rows after the filter: kcat random_splits_grouped_sequence train 49,781 / val 2,794 / test 2,795.
All outputs go to `--work_dir` (default `outputs/baselines/UniKP`). The only other write is a missing ProtT5 vector in the CataPro cache folder (none was missing). Nothing is written to the split folders.

## 4. Files
Changed: `common.py`, `feature_pipeline.py`, `cache_embeddings.py`, `build_split_features.py`, `train_single_target_tvt.py`, `launch_parallel_retrain_original.py` (adds `--dry_run`, `--protein_cache_dir`), `README.md`, `commands.txt`, `REPORT.md`.
Unchanged: `aggregate_unikp_results.py`.
Added: `verify_equivalence.sh`, `tests/test_original_equivalence.py`, `tests/conftest.py`, `tests/fixtures/tiny_task_table.csv` (175 table rows with short synthetic sequences), `tests/fixtures/tiny_t5_tokenizer/spiece.model` (tiny sentencepiece model, copied from the CataPro tests).
Removed: the UniKP ProtT5 cache code (`protein_cache_key`, `protein_cache_path`, `normalize_sequence`, the `.npz` protein files).
Original UniKP files outside `emulator_bench` are unchanged.

## 5. Bugs fixed
- `DEFAULT_SPLIT_GROUPS` missed `enzyme_structure_splits` and `conformer_cosine_splits`. Now 7 groups.
- `ROW_KEY_COLUMNS` named `pH` and `Temperature`, which the task tables no longer have. The key now has 10 columns.
- The SMILES vectors were first built on the GPU. They are now built on the CPU and match the original exactly. `CACHE_VERSION` is 5, so the cache folder is `smiles_trfm_v5_<fingerprint>` and the feature matrices were rebuilt. No GPU vector is reused.
- A split row missing from the task table silently used column 0. It now raises an error. All split rows of all 42 split folders match (the feature build finished without error).
- Protein vectors now come from the shared CataPro cache (same model, text, fp32 path), so the 30,209 sequences are not embedded twice.
- Sequence text is no longer upper-cased or stripped. The original does not do it, and the data has no such change.

## 6. Deviations, and why behaviour is unchanged
- ProtT5 runs in length-sorted batches of 8 with an attention mask, and the mean is fp32 on the GPU. The original runs one sequence at a time with a float64 python-loop mean. Measured difference: see section 9. It is float rounding.
- The SMILES Transformer runs on the CPU in batches of 512 (original: CPU, chunks of 100). Inputs are padded to 220 tokens, so rows do not interact. The vectors are bit-exact (section 9). An earlier GPU build differed by up to 2.8e-5; it was replaced (cache version 5).
- `random_state=seed` is set so seeds differ in a repeatable way. The original is unseeded.
- `n_jobs` is a speed control. sklearn draws the tree seeds before the parallel fit, so every tree is identical for any `n_jobs` (tested on every tree array). The summed prediction differs by float64 rounding (max 2.2e-15 between a 48-thread and a 12-thread run), because parallel predict adds trees in thread order.
- Feature matrices are written without compression (same values, faster).
- The multi-component SMILES filter applies to train only, and only for kcat, so all baselines are tested on the same test rows.

## 7. Commands
From `EMULaToR/baselines/UniKP`, env mldb (`PY=python`):
```
export CUDA_VISIBLE_DEVICES=0
$PY emulator_bench/launch_parallel_retrain_original.py --dry_run
$PY emulator_bench/launch_parallel_retrain_original.py --tasks kcat --drop_dot_smiles_train --jobs 4 --model_n_jobs 12
$PY emulator_bench/launch_parallel_retrain_original.py --tasks km --jobs 4 --model_n_jobs 12
$PY emulator_bench/aggregate_unikp_results.py
bash emulator_bench/verify_equivalence.sh --gpu
```
`--jobs 4 --model_n_jobs 12` uses the 48 threads of this machine.

Measured timings (kcat random_splits_grouped_sequence, train 49,781 x 2048, seed 666):

| Step | Result |
|---|---|
| SMILES Transformer cache, CPU (48 threads) | kcat 82 s (41,641 pairs), km 113 s (57,541 pairs); peak RAM 2.5 / 2.6 GB. The earlier GPU build took 22 s / 29 s. |
| ProtT5 cache | complete, 0 vectors computed (30,209 texts from the CataPro cache) |
| Feature build, all 7 split folders of a task in one process | kcat 32 s (21 matrices, 3.3 GB), km 45 s (4.6 GB); peak RAM 3.7 GB / 4.6 GB |
| Old style (one train split, per-row loads, compressed write) | 5.0 s assemble + 6.6 s write; the new matrix is identical (`np.array_equal`) |
| Matrix load in the trainer, compressed vs not | 0.89 s vs 0.15 s |
| 1 run, `--model_n_jobs 48` | fit 88.6 s, predict + write 0.55 s, dump 0.13 s, `model.joblib` 642 MB, peak RSS 1.36 GB |
| 4 runs at once x 12 threads | fit 294-299 s each, 74.5 s per run overall, peak RSS 1.24 GB each |
| 5 runs at once x 10 threads | fit 360-364 s each, 72.4 s per run overall |

Fit time per tree: 100 trees need 3 waves at 48 threads (29.5 s per wave) and 9 waves at 12 threads (33 s per wave). One tree takes about 30 to 33 s on one hyper-thread when all 48 threads are busy. One 48-thread run wastes the tail wave (4 trees in the last wave), so 4 x 12 is 16 percent faster in total. 5 x 10 gives only 3 percent more. km was not timed. Estimate: train rows grow 1.4 times, so about 110 s per run and about 65 minutes for all 42 runs (kcat 26 min, km 39 min). `model.joblib` is 0.64 GB for kcat; expect about 0.9 GB for km, about 30 GB in total.
Predictions at 48 and 12 threads for seed 666 differ by at most 2.2e-15. Test RMSE is 1.1173 in both runs.

## 8. Smoke test
`launch_parallel_retrain_original.py --max_rows_per_split 64 --seeds 666 --split_groups random_splits_grouped_sequence enzyme_structure_splits --jobs 2 --model_n_jobs 4` (GPU 0, both tasks): cache, feature, train and aggregate stages finished, 4 runs.
Real data: caches and matrices for both tasks are built under `outputs/baselines/UniKP`. The 42 training runs were not started. `--dry_run` lists all 42 runs.

## 9. Equivalence verdict
Verdict: EQUIVALENT-EXCEPT-DOCUMENTED. The documented deviations are float rounding and the seeded `random_state`.

`bash emulator_bench/verify_equivalence.sh` (CPU, 9 tests, 0 skipped) and `--gpu` (2 more tests) all pass after the CPU rebuild. The tests call the original `smiles_to_vec`, `Seq_to_vec` and `ExtraTreesRegressor()`.

| Check | Result |
|---|---|
| SMILES vectors, 175 fixture rows (crosses the chunk boundary at 100, includes dots and a 259-char SMILES), CPU, batch 512 | exact equality (`assert_array_equal`) |
| Protein text (>1000 aa truncation, U/Z/O/B to X) | identical to the text the original passes to the tokenizer |
| Protein vectors, tiny random T5, CPU, original one-at-a-time vs wrapper batched | within rtol 1e-4, atol 1e-5 (the test tolerance) |
| Feature assembly through the wrapper CLIs, kcat rule (drop dot SMILES from train) and km rule (keep all) | SMILES block exact, protein block within the same tolerance, targets exact, column order SMILES then protein, val/test keep dot rows |
| Row key and missing row | shuffled split rows get `row % 100`; a changed row raises an error |
| Extra Trees parameters | equal to the `ExtraTreesRegressor()` defaults except `n_jobs` and `random_state` |
| Extra Trees, same seed | every tree array identical to a direct `ExtraTreesRegressor(random_state=666)` fit for `n_jobs` 1 and 8; predictions equal within 1e-12; the training script gives the same predictions as a direct fit |
| GPU: real ProtT5, original one-at-a-time fp32 path vs CataPro cache, 35 sequences (7 to 7,096 aa, 11 longer than 1000 aa, includes the 6 longest) | max abs diff 4.1e-7, max rel diff 2.5e-4 (components with abs value below 1e-3 use that floor), mean abs diff 1.4e-8 |
| Rebuilt real SMILES cache (CPU) vs original `smiles_to_vec`, first 250 kcat table rows (`--gpu` mode) | exact, max abs diff 0.0 |
| Earlier GPU SMILES build vs original (replaced) | max abs diff 2.8e-5, mean 4.0e-7 |

Downstream noise check, run before the CPU rebuild, so the "cached" arm used the GPU SMILES vectors (kcat random_splits_grouped_sequence; train subsample of 8,000 rows plus all 2,795 test rows; 6,919 sequences embedded with the original `Seq_to_vec` one at a time on the GPU in 12 minutes; Extra Trees, seeds 666 to 673):

| Arm | Test RMSE, mean over 8 seeds (std) | R2 | Pearson | Paired RMSE gap to cached: mean, standard error, max over seeds |
|---|---|---|---|---|
| cached protein + GPU SMILES (the wrapper) | 1.3048 (0.0035) | 0.3408 | 0.5852 | |
| exact original protein, GPU SMILES | 1.3064 (0.0041) | 0.3393 | 0.5837 | +0.0015, 0.0021, 0.0099 |
| exact original protein and exact CPU SMILES | 1.3045 (0.0040) | 0.3411 | 0.5855 | -0.0003, 0.0007, 0.0033 |

Vector differences on the subsample: protein max abs 2.1e-6 (max rel 8.4e-4 with the 1e-3 floor); SMILES GPU vs CPU max abs 3.0e-5.
Verdict: the exact-vs-cached gap cannot be told apart from seed noise. The mean paired gap is within one standard error of zero. The seed-to-seed std of the cached arm is 0.0035 RMSE. A single seed can move by up to 0.01 RMSE under any tiny feature change (a 1e-6 relative jitter does the same), because fully grown trees re-roll. The wrapper features are accepted.
A first 3-seed run (666, 777, 888) of the second arm gave gaps of +0.010, +0.014 and 0.000 RMSE. This is the same noise. The 8-seed table is the better estimate.

## 10. Open issues
- The old GPU SMILES cache folders (`outputs/baselines/UniKP/embeddings/smiles_trfm_v4_*`, 331 MB and 459 MB) are not used any more and were not deleted.
- The default `--protein_cache_dir` points into the CataPro output folder. Another machine must build this cache (about 30k sequences on a GPU) or pass its own `--protein_cache_dir`.
- The row key relies on 10 columns being unique. The wrapper checks this and stops if not.
- The sklearn version (1.8.0) is newer than the paper (1.1.1). Defaults are the same for the used parameters.
- km timings are estimates, not measurements.
- Paper ablation rows for UniKP show std 0.00 for some axes. The std on an 8k subsample here is 0.0035 RMSE. The full data was not rerun for this.
- Predictions can differ in the last float64 bit for different `--model_n_jobs` (max 2.2e-15). The trees are the same.
