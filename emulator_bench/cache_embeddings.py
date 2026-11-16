import argparse
import time
from pathlib import Path

import pandas as pd
import torch

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    CACHE_VERSION,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PROTEIN_CACHE_DIR,
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_WORK_DIR,
    TASKS,
    atomic_save_json,
    discover_split_jobs,
    normalize_smiles,
    prepare_frame,
    TaskTable,
    read_table,
    smiles_cache_path,
)
from emulator_bench.feature_pipeline import (
    PROT_MODEL_ID,
    iter_protein_embeddings,
    iter_smiles_embeddings,
    protein_cache_file,
    save_npy_atomic,
    save_npz_atomic,
    unikp_sequence_text,
)


def collect_unique_values(jobs, args, tables):
    texts = set()
    pairs = {task: set() for task in tables}
    for job in jobs:
        for split in ("train", "val", "test"):
            path = Path(job[f"{split}_path"])
            frame = prepare_frame(read_table(path), split, args, path, tables[job["task"]])
            texts.update(unikp_sequence_text(value) for value in frame[args.sequence_col].astype(str).unique())
            pairs[job["task"]].update(zip((normalize_smiles(value) for value in frame[args.smiles_col].astype(str)), frame["trfm_column"]))
    return sorted(texts), pairs


def cache_proteins(args, texts):
    pending = [text for text in texts if args.overwrite or not protein_cache_file(args.protein_cache_dir, text).exists()]
    if not pending:
        print("Protein cache is already complete.", flush=True)
        return {"total": len(texts), "written": 0}

    written = 0
    for text, vector in iter_protein_embeddings(pending, device=torch.device(args.device), batch_size=args.protein_batch_size):
        save_npy_atomic(protein_cache_file(args.protein_cache_dir, text), vector)
        written += 1
    return {"total": len(texts), "written": written}


def cache_smiles(args, pairs, fingerprint):
    pending = [pair for pair in pairs if args.overwrite or not smiles_cache_path(args.embeddings_dir, *pair, fingerprint).exists()]
    if not pending:
        print("SMILES cache is already complete.", flush=True)
        return {"total": len(pairs), "written": 0}

    device = torch.device(args.smiles_device)
    embedded = iter_smiles_embeddings(pending, device=device, batch_size=args.smiles_batch_size)
    written = 0
    for (smiles, column), vector in embedded:
        save_npz_atomic(
            smiles_cache_path(args.embeddings_dir, smiles, column, fingerprint),
            {
                "vector": vector,
                "length": pd.Series([len(smiles)], dtype="int32").to_numpy(),
            },
        )
        written += 1
    return {"total": len(pairs), "written": written}


def main():
    parser = argparse.ArgumentParser(description="Cache UniKP ProtT5 and SMILES Transformer embeddings for the splits of the given tasks.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for missing ProtT5 vectors.")
    parser.add_argument("--smiles_device", type=str, default="cpu", help="CPU matches the original smiles_to_vec bit for bit.")
    parser.add_argument("--protein_cache_dir", type=str, default=str(DEFAULT_PROTEIN_CACHE_DIR), help="ProtT5 cache shared with the CataPro wrapper. Missing vectors are added to it.")
    parser.add_argument("--protein_batch_size", type=int, default=8, help="Same as the CataPro wrapper, so new vectors match its cache.")
    parser.add_argument("--smiles_batch_size", type=int, default=512)
    parser.add_argument("--max_rows_per_split", type=int, default=None, help="Smoke tests only.")
    parser.add_argument("--drop_dot_smiles_train", action="store_true", help="Drop multi-component SMILES from the train split (original kcat script).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.embeddings_dir = Path(args.work_dir) / "embeddings"
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)
    args.protein_cache_dir = Path(args.protein_cache_dir)

    jobs = [
        job
        for task in args.tasks
        for job in ({**job, "task": task} for job in discover_split_jobs(Path(args.dataset_root) / task, split_groups=args.split_groups))
    ]
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.dataset_root}")

    started = time.time()
    tables = {task: TaskTable(args.dataset_root, task) for task in args.tasks}
    texts, pairs = collect_unique_values(jobs, args, tables)
    print(f"Discovered {len(jobs)} split jobs", flush=True)
    print(f"Unique processed protein texts: {len(texts)}", flush=True)
    print(f"Unique (SMILES, column) pairs: {sum(len(values) for values in pairs.values())}", flush=True)

    protein_stats = cache_proteins(args, texts)
    smiles_stats = {task: cache_smiles(args, sorted(pairs[task]), tables[task].fingerprint) for task in args.tasks}
    manifest = {
        "cache_version": CACHE_VERSION,
        "table_fingerprints": {task: table.fingerprint for task, table in tables.items()},
        "dataset_root": str(args.dataset_root),
        "tasks": args.tasks,
        "protein_model": PROT_MODEL_ID,
        "protein_cache_dir": str(args.protein_cache_dir),
        "smiles_model": "UniKP repository trfm_12_23000.pkl",
        "split_groups": list(args.split_groups),
        "protein_cache": protein_stats,
        "smiles_cache": smiles_stats,
        "elapsed_seconds": time.time() - started,
    }
    atomic_save_json(args.embeddings_dir / f"manifest_{'_'.join(args.tasks)}.json", manifest)
    print(f"Saved cache manifest to {args.embeddings_dir}", flush=True)


if __name__ == "__main__":
    main()
