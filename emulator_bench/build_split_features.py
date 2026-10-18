import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_WORK_DIR,
    TASKS,
    atomic_save_csv,
    discover_split_jobs,
    feature_job_dir,
    normalize_sequence,
    prepare_frame,
    split_fingerprint,
    protein_cache_path,
    read_table,
    smiles_cache_path,
    TaskTable,
)
from emulator_bench.feature_pipeline import load_cached_vector, save_npz_atomic


META_COLUMNS = ["sequence", "smiles", "value", "log10_value", "smiles_hash", "uniprot_date"]


def build_one_split(frame: pd.DataFrame, args, out_dir: Path, split_name: str, source_path: Path, table: TaskTable) -> dict:
    matrix_path = out_dir / f"{split_name}.npz"
    meta_path = out_dir / f"{split_name}_metadata.csv"
    fingerprint = split_fingerprint(source_path, split_name, args, table.fingerprint)
    if matrix_path.exists() and meta_path.exists() and not args.overwrite:
        with np.load(matrix_path, allow_pickle=False) as data:
            current = "fingerprint" in data.files and str(data["fingerprint"]) == fingerprint
        if current:
            return {"split": split_name, "status": "skipped_exists", "rows": int(len(frame))}

    vectors = {}

    def cached(path: Path) -> np.ndarray:
        if path not in vectors:
            vectors[path] = load_cached_vector(path)
        return vectors[path]

    x_array = np.zeros((len(frame), 2048), dtype=np.float32)
    for row_index, (sequence, smiles, column) in enumerate(zip(frame[args.sequence_col], frame[args.smiles_col], frame["trfm_column"])):
        smiles_vec = cached(smiles_cache_path(args.embeddings_dir, smiles, column, table.fingerprint))
        protein_vec = cached(protein_cache_path(args.embeddings_dir, normalize_sequence(sequence)))
        x_array[row_index] = np.concatenate([smiles_vec, protein_vec])
    y_array = frame[args.target_col].to_numpy(dtype=np.float64)

    save_npz_atomic(matrix_path, {"X": x_array, "y": y_array, "fingerprint": np.array(fingerprint)})
    meta = frame[[c for c in META_COLUMNS if c in frame.columns]].copy()
    meta.insert(0, "source_row", range(len(meta)))
    meta.insert(0, "source_path", str(source_path))
    atomic_save_csv(meta_path, meta)
    return {"split": split_name, "status": "written", "rows": int(len(frame))}


def main():
    parser = argparse.ArgumentParser(description="Materialize UniKP train/val/test feature matrices from cached embeddings.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--max_rows_per_split", type=int, default=None, help="Smoke tests only.")
    parser.add_argument("--drop_dot_smiles_train", action="store_true", help="Drop multi-component SMILES from the train split (original kcat script).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.embeddings_dir = Path(args.work_dir) / "embeddings"
    started = time.time()
    for task in args.tasks:
        table = TaskTable(args.dataset_root, task)
        jobs = discover_split_jobs(Path(args.dataset_root) / task, split_groups=args.split_groups)
        if not jobs:
            raise FileNotFoundError(f"No split jobs discovered for {task} in {args.dataset_root}")
        feature_root = Path(args.work_dir) / task / "feature_matrices"
        summary_rows = []
        for job in jobs:
            out_dir = feature_job_dir(feature_root, job["split_group"], job["split_name"])
            out_dir.mkdir(parents=True, exist_ok=True)
            for split_name in ("train", "val", "test"):
                path = Path(job[f"{split_name}_path"])
                frame = prepare_frame(read_table(path), split_name, args, path, table)
                row = build_one_split(frame, args, out_dir, split_name, path, table)
                summary_rows.append({"split_group": job["split_group"], "split_name": job["split_name"], **row})
        atomic_save_csv(feature_root / "feature_matrix_index.csv", pd.DataFrame(summary_rows))
        print(f"[{task}] saved feature matrix index to {feature_root}", flush=True)
    print(f"Elapsed seconds: {time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
