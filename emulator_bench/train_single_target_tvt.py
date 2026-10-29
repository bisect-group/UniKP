import argparse
import json
import resource
import time
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_SEEDS,
    atomic_save_csv,
    atomic_save_json,
    configure_cpu_environment,
    regression_metrics,
    set_seed,
)


def load_feature_split(feature_dir: Path, split: str):
    matrix_path = Path(feature_dir) / f"{split}.npz"
    metadata_path = Path(feature_dir) / f"{split}_metadata.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {matrix_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature metadata: {metadata_path}")
    with np.load(matrix_path, allow_pickle=False) as data:
        x = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.float64, copy=False)
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != len(y):
        raise ValueError(f"Metadata/target length mismatch for {split}: {len(metadata)} != {len(y)}")
    return x, y, metadata


def save_predictions(path: Path, metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    frame = metadata.copy()
    frame["y_true"] = y_true
    frame["y_pred"] = y_pred
    atomic_save_csv(path, frame)


def make_model(seed: int, n_jobs):
    """The original code calls ExtraTreesRegressor() with sklearn defaults; only n_jobs and the seed are set here."""
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(n_jobs=n_jobs, random_state=seed)


def maybe_threadpool_limits(model_n_jobs):
    if model_n_jobs is None or int(model_n_jobs) <= 0:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=int(model_n_jobs))
    except Exception:
        return nullcontext()


def main():
    parser = argparse.ArgumentParser(description="Train one UniKP ExtraTrees model on cached train/val/test feature matrices.")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--model_n_jobs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    configure_cpu_environment(args.model_n_jobs)
    set_seed(args.seed)
    feature_dir = Path(args.feature_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_path = out_dir / "final_results_test.csv"
    model_path = out_dir / "model.joblib"
    if done_path.exists() and model_path.exists() and not args.overwrite:
        print(f"Skipping completed run: {out_dir}", flush=True)
        return

    x_train, y_train, meta_train = load_feature_split(feature_dir, "train")
    x_val, y_val, meta_val = load_feature_split(feature_dir, "val")
    x_test, y_test, meta_test = load_feature_split(feature_dir, "test")
    if len(y_train) == 0:
        raise ValueError(f"Training split is empty after filtering: {feature_dir}")

    model = make_model(args.seed, args.model_n_jobs)
    started = time.time()
    with maybe_threadpool_limits(args.model_n_jobs):
        model.fit(x_train, y_train)
    fit_seconds = time.time() - started

    started = time.time()
    joblib.dump(model, model_path)
    dump_seconds = time.time() - started
    summary = {
        "fit_seconds": fit_seconds,
        "dump_seconds": dump_seconds,
        "model_bytes": model_path.stat().st_size,
        "feature_dir": str(feature_dir),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "model_n_jobs": args.model_n_jobs,
        "extra_trees_params": json.dumps(model.get_params(), sort_keys=True),
    }
    started = time.time()
    for split_name, x, y, meta in (
        ("train", x_train, y_train, meta_train),
        ("val", x_val, y_val, meta_val),
        ("test", x_test, y_test, meta_test),
    ):
        pred = model.predict(x)
        metrics = regression_metrics(y, pred)
        metrics.update({"split": split_name, "rows": int(len(y)), "seed": args.seed})
        atomic_save_csv(out_dir / f"final_results_{split_name}.csv", pd.DataFrame([metrics]))
        save_predictions(out_dir / f"predictions_{split_name}.csv", meta, y, pred)
        summary.update({f"{split_name}_{key}": value for key, value in metrics.items() if key != "split"})
    summary["predict_seconds"] = time.time() - started
    summary["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    atomic_save_json(out_dir / "run_summary.json", summary)
    print(f"Saved run to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
