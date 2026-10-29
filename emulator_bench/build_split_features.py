import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_PROTEIN_CACHE_DIR,
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_WORK_DIR,
    TASKS,
    atomic_save_csv,
    discover_split_jobs,
    feature_job_dir,
    prepare_frame,
    split_fingerprint,
    read_table,
    smiles_cache_path,
    TaskTable,
)
from emulator_bench.feature_pipeline import load_cached_vector, protein_cache_file, save_npz_atomic, unikp_sequence_text


META_COLUMNS = ["sequence", "smiles", "value", "log10_value", "smiles_hash", "uniprot_date"]


def is_current(matrix_path: Path, meta_path: Path, fingerprint: str) -> bool:
    if not (matrix_path.exists() and meta_path.exists()):
        return False
    with np.load(matrix_path, allow_pickle=False) as data:
        return "fingerprint" in data.files and str(data["fingerprint"]) == fingerprint


def load_vectors(paths, loader) -> tuple:
    """Load each unique vector once. Returns (matrix, path -> row)."""
    unique = list(dict.fromkeys(paths))
    with ThreadPoolExecutor(max_workers=16) as pool:
        vectors = list(pool.map(loader, unique))
    return np.stack(vectors).astype(np.float32, copy=False), {path: idx for idx, path in enumerate(unique)}


def build_task(task: str, args) -> None:
    table = TaskTable(args.dataset_root, task)
    jobs = discover_split_jobs(Path(args.dataset_root) / task, split_groups=args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered for {task} in {args.dataset_root}")
    feature_root = Path(args.work_dir) / task / "feature_matrices"

    pending, summary_rows = [], []
    for job in jobs:
        out_dir = feature_job_dir(feature_root, job["split_group"], job["split_name"])
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val", "test"):
            path = Path(job[f"{split_name}_path"])
            frame = prepare_frame(read_table(path), split_name, args, path, table)
            fingerprint = split_fingerprint(path, split_name, args, table.fingerprint)
            matrix_path, meta_path = out_dir / f"{split_name}.npz", out_dir / f"{split_name}_metadata.csv"
            status = "skipped_exists" if not args.overwrite and is_current(matrix_path, meta_path, fingerprint) else "written"
            summary_rows.append({"split_group": job["split_group"], "split_name": job["split_name"], "split": split_name, "status": status, "rows": int(len(frame))})
            if status == "written":
                pending.append((frame, path, split_name, matrix_path, meta_path, fingerprint))

    if pending:
        frames = [item[0] for item in pending]
        smiles_paths = [[smiles_cache_path(args.embeddings_dir, smiles, column, table.fingerprint) for smiles, column in zip(frame[args.smiles_col], frame["trfm_column"])] for frame in frames]
        protein_paths = [[protein_cache_file(args.protein_cache_dir, unikp_sequence_text(sequence)) for sequence in frame[args.sequence_col]] for frame in frames]
        smiles_matrix, smiles_rows = load_vectors((p for paths in smiles_paths for p in paths), load_cached_vector)
        protein_matrix, protein_rows = load_vectors((p for paths in protein_paths for p in paths), np.load)
        for (frame, path, split_name, matrix_path, meta_path, fingerprint), s_paths, p_paths in zip(pending, smiles_paths, protein_paths):
            x_array = np.concatenate([smiles_matrix[[smiles_rows[p] for p in s_paths]], protein_matrix[[protein_rows[p] for p in p_paths]]], axis=1)
            save_npz_atomic(matrix_path, {"X": x_array, "y": frame[args.target_col].to_numpy(dtype=np.float64), "fingerprint": np.array(fingerprint)})
            meta = frame[[c for c in META_COLUMNS if c in frame.columns]].copy()
            meta.insert(0, "source_row", range(len(meta)))
            meta.insert(0, "source_path", str(path))
            atomic_save_csv(meta_path, meta)
    atomic_save_csv(feature_root / "feature_matrix_index.csv", pd.DataFrame(summary_rows))
    print(f"[{task}] saved feature matrix index to {feature_root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Materialize UniKP train/val/test feature matrices from cached embeddings.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--protein_cache_dir", type=str, default=str(DEFAULT_PROTEIN_CACHE_DIR))
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--max_rows_per_split", type=int, default=None, help="Smoke tests only.")
    parser.add_argument("--drop_dot_smiles_train", action="store_true", help="Drop multi-component SMILES from the train split (original kcat script).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.embeddings_dir = Path(args.work_dir) / "embeddings"
    args.protein_cache_dir = Path(args.protein_cache_dir)
    started = time.time()
    for task in args.tasks:
        build_task(task, args)
    print(f"Elapsed seconds: {time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
