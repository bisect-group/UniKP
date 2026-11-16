import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_WORK_DIR, TASKS, atomic_save_csv, summarize_seed_runs


def collect_runs(work_dir: Path, task: str):
    """Read `<work_dir>/<task>/<split_group>/<split_name>/seed_<seed>` run folders."""
    root = Path(work_dir) / task
    rows = []
    for test_path in sorted(root.glob("*/*/seed_*/final_results_test.csv")):
        run_dir = test_path.parent
        row = {"task": task, "split_group": run_dir.parts[-3], "split_name": run_dir.parts[-2], "seed": int(run_dir.name.removeprefix("seed_"))}
        for split in ("train", "val", "test"):
            metrics_path = run_dir / f"final_results_{split}.csv"
            if metrics_path.exists():
                row.update({f"{split}_{key}": value for key, value in pd.read_csv(metrics_path).iloc[0].items()})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate UniKP emulator_bench results over seeds.")
    parser.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: <work_dir>/aggregated.")
    args = parser.parse_args()

    rows = [row for task in args.tasks for row in collect_runs(Path(args.work_dir), task)]
    if not rows:
        raise FileNotFoundError(f"No finished runs under {args.work_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.work_dir) / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [col for col in pd.DataFrame(rows).columns if col.startswith("test_") and col not in {"test_split", "test_seed"}]
    atomic_save_csv(out_dir / "aggregated_run_summaries.csv", pd.DataFrame(rows))
    atomic_save_csv(out_dir / "summary_by_task_split.csv", summarize_seed_runs(rows, ["task", "split_group", "split_name"], metric_cols))
    print(f"Saved aggregation outputs to {out_dir}")


if __name__ == "__main__":
    main()
