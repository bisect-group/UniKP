import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_WORK_DIR,
    TASKS,
    atomic_save_csv,
    discover_split_jobs,
    feature_job_dir,
    summarize_seed_runs,
)


BENCH_DIR = REPO_ROOT / "emulator_bench"


def run_stage(script, args, extra=()):
    cmd = [
        sys.executable,
        str(BENCH_DIR / script),
        "--tasks", *args.tasks,
        "--dataset_root", args.dataset_root,
        "--work_dir", args.work_dir,
        "--split_groups", *args.split_groups,
        *extra,
    ]
    if args.max_rows_per_split is not None:
        cmd.extend(["--max_rows_per_split", str(args.max_rows_per_split)])
    if args.drop_dot_smiles_train:
        cmd.append("--drop_dot_smiles_train")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def build_experiments(args):
    experiments = []
    for task in args.tasks:
        for job in discover_split_jobs(Path(args.dataset_root) / task, split_groups=args.split_groups):
            fdir = feature_job_dir(Path(args.work_dir) / task / "feature_matrices", job["split_group"], job["split_name"])
            for seed in args.seeds:
                out_dir = Path(args.work_dir) / task / "results" / job["split_group"] / job["split_name"] / f"seed_{seed}"
                experiments.append(
                    {"task": task, "split_group": job["split_group"], "split_name": job["split_name"], "feature_dir": fdir, "out_dir": out_dir, "seed": int(seed)}
                )
    return experiments


def run_one(exp, args):
    cmd = [
        sys.executable,
        str(BENCH_DIR / "train_single_target_tvt.py"),
        "--feature_dir", str(exp["feature_dir"]),
        "--out_dir", str(exp["out_dir"]),
        "--seed", str(exp["seed"]),
    ]
    env = os.environ.copy()
    if args.model_n_jobs is not None:
        cmd.extend(["--model_n_jobs", str(args.model_n_jobs)])
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[key] = str(args.model_n_jobs)
    if args.overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def run_parallel(experiments, args):
    work_queue = queue.Queue()
    for exp in experiments:
        work_queue.put(exp)
    rows = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                exp = work_queue.get_nowait()
            except queue.Empty:
                return
            row = {k: exp[k] for k in ("task", "split_group", "split_name", "seed")}
            row["out_dir"] = str(exp["out_dir"])
            try:
                run_one(exp, args)
                row["status"] = "ok"
            except Exception as exc:
                row.update(status="failed", error=str(exc))
            with lock:
                rows.append(row)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.num_workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return rows


def collect_metrics(rows):
    for row in rows:
        for split in ("train", "val", "test"):
            path = Path(row["out_dir"]) / f"final_results_{split}.csv"
            if path.exists():
                row.update({f"{split}_{key}": value for key, value in pd.read_csv(path).iloc[0].items()})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Multi-seed UniKP evaluation with the original ExtraTrees settings on EMULaToR splits.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Folder with kcat/ and km/ split folders.")
    parser.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR), help="Output folder for embeddings, feature matrices, and results.")
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--model_n_jobs", type=int, default=None)
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--drop_dot_smiles_train", action="store_true", help="Drop multi-component SMILES from train (original kcat script only).")
    parser.add_argument("--max_rows_per_split", type=int, default=None, help="Smoke tests only.")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--skip_feature_build", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.skip_cache:
        run_stage("cache_embeddings.py", args, ["--device", args.cache_device])
    if not args.skip_feature_build:
        run_stage("build_split_features.py", args, ["--overwrite"] if args.overwrite else [])
    experiments = build_experiments(args)
    if not experiments:
        raise FileNotFoundError(f"No split jobs discovered in {args.dataset_root}")
    rows = collect_metrics(run_parallel(experiments, args))

    frame = pd.DataFrame(rows)
    for task in args.tasks:
        task_rows = [row for row in rows if row["task"] == task]
        root = Path(args.work_dir) / task / "results"
        root.mkdir(parents=True, exist_ok=True)
        metric_cols = [col for col in frame.columns if col.startswith("test_")]
        atomic_save_csv(root / "retrain_summary_runs.csv", pd.DataFrame(task_rows))
        summarize_seed_runs(task_rows, ["split_group", "split_name"], metric_cols).to_csv(root / "retrain_summary_by_split.csv", index=False)
    failed = frame[frame["status"] == "failed"] if "status" in frame else frame.iloc[0:0]
    if len(failed):
        raise SystemExit(f"{len(failed)} runs failed:\n{failed[['task', 'split_group', 'split_name', 'seed']].to_string()}")


if __name__ == "__main__":
    main()
