import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EMULATOR_ROOT = REPO_ROOT.parents[1]
DEFAULT_DATASET_ROOT = EMULATOR_ROOT / "data" / "processed" / "datasets" / "kinetic_dataset"
DEFAULT_WORK_DIR = EMULATOR_ROOT / "outputs" / "baselines" / "UniKP"
TASKS = ["kcat", "km"]
CACHE_VERSION = 3
PROT_MODEL_ID = "Rostlab/prot_t5_xl_uniref50"
TRFM_CHUNK = 100
ROW_KEY_COLUMNS = ["ECNumber", "uniprot_id", "sequence", "smiles", "value", "Organism", "pH", "Temperature", "mutations", "substrate", "source", "enzymeType"]
DEFAULT_SEEDS = [666, 777, 888]
DEFAULT_SPLIT_GROUPS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "substrate_splits",
]


def stable_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().upper().split())


def normalize_smiles(smiles: str) -> str:
    return str(smiles).strip()


def protein_cache_key(sequence: str) -> str:
    return stable_hash(f"{PROT_MODEL_ID}\t{normalize_sequence(sequence)}")


def protein_cache_path(embeddings_dir: Path, sequence: str) -> Path:
    key = protein_cache_key(sequence)
    return Path(embeddings_dir) / f"proteins_v{CACHE_VERSION}" / key[:2] / f"{key}.npz"


def smiles_cache_path(embeddings_dir: Path, smiles: str, column: int, fingerprint: str) -> Path:
    """One vector per (SMILES, chunk column); the folder name holds the table fingerprint."""
    key = stable_hash(f"{normalize_smiles(smiles)}\t{column}")
    return Path(embeddings_dir) / f"smiles_trfm_v{CACHE_VERSION}_{fingerprint}" / key[:2] / f"{key}.npz"


def table_path(dataset_root: Path, task: str) -> Path:
    return Path(dataset_root) / task / f"{task}_kinetic_params_1d.parquet"


def row_keys(frame: pd.DataFrame) -> pd.Series:
    return frame[ROW_KEY_COLUMNS].astype(str).agg("\x1f".join, axis=1)


class TaskTable:
    """The full unsplit task table. The original UniKP encodes it in chunks of 100 rows.

    Its positional encoding is added along the batch axis, so row r gets the vector of chunk column r % 100.
    """

    def __init__(self, dataset_root: Path, task: str):
        path = table_path(dataset_root, task)
        frame = pd.read_parquet(path, columns=ROW_KEY_COLUMNS)
        keys = row_keys(frame)
        if keys.duplicated().any():
            raise ValueError(f"Row key is not unique in {path}")
        self.columns = dict(zip(keys, np.arange(len(keys)) % TRFM_CHUNK))
        self.fingerprint = stable_hash("\n".join(frame["smiles"].astype(str)))[:12]

    def attach_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add `trfm_column`. Rows missing from the table use column 0 (the count is printed)."""
        columns = row_keys(frame).map(self.columns)
        if columns.isna().any():
            print(f"WARNING: {int(columns.isna().sum())} of {len(frame)} split rows are not in the task table; using column 0.", flush=True)
        return frame.assign(trfm_column=columns.fillna(0).astype(int))


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def atomic_save_json(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def atomic_save_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _find_split_file(directory: Path, stem: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def discover_split_jobs(base_dir: Path, split_groups: Optional[Iterable[str]] = None) -> List[Dict[str, str]]:
    """Find `<group>/{train,val,test}` or `<group>/threshold_*/{train,val,test}` folders."""
    jobs: List[Dict[str, str]] = []
    for split_group in split_groups or DEFAULT_SPLIT_GROUPS:
        group_dir = Path(base_dir) / split_group
        if not group_dir.exists():
            continue
        candidates = [("all", group_dir)]
        if not (group_dir / "train.parquet").exists():
            candidates = [(child.name, child) for child in sorted(group_dir.glob("threshold_*")) if child.is_dir()]
        for split_name, directory in candidates:
            paths = {name: _find_split_file(directory, name) for name in ("train", "val", "test")}
            if all(paths.values()):
                jobs.append({"split_group": split_group, "split_name": split_name, **{f"{k}_path": str(v) for k, v in paths.items()}})
    return jobs


def feature_job_dir(feature_root: Path, split_group: str, split_name: str) -> Path:
    return Path(feature_root) / split_group / split_name


def split_fingerprint(path: Path, split: str, args, table_fingerprint: str) -> str:
    """Hash of the inputs and flags that determine a feature matrix."""
    stat = Path(path).stat()
    drop_dot = bool(args.drop_dot_smiles_train) and split == "train"
    key = [str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, args.sequence_col, args.smiles_col, args.target_col, args.max_rows_per_split, drop_dot, table_fingerprint, CACHE_VERSION]
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()


def prepare_frame(frame: pd.DataFrame, split: str, args, path: Path, table: TaskTable) -> pd.DataFrame:
    """Add the chunk column, then apply row cap (smoke tests) and the optional multi-component SMILES filter to the train split."""
    require_columns(frame, [args.sequence_col, args.smiles_col, args.target_col, *ROW_KEY_COLUMNS], path)
    frame = table.attach_columns(frame)
    if args.max_rows_per_split is not None:
        frame = frame.head(int(args.max_rows_per_split))
    if split == "train" and args.drop_dot_smiles_train:
        frame = frame[~frame[args.smiles_col].astype(str).str.contains(".", regex=False)]
    return frame[pd.notna(frame[args.target_col])].reset_index(drop=True)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2_score": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
    residual = y_true - y_pred
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    true_rank = pd.Series(y_true).rank(method="average").to_numpy()
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()
    if y_true.size < 2 or np.std(true_rank) == 0 or np.std(pred_rank) == 0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1])
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2_score": 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "pearson": pearson,
        "spearman": spearman,
    }


def summarize_seed_runs(rows: List[Dict], group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    present_metrics = []
    for col in metric_cols:
        if col not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().any():
            frame[col] = numeric
            present_metrics.append(col)
    if not present_metrics:
        return frame[group_cols].drop_duplicates()
    grouped = frame.groupby(group_cols, dropna=False)[present_metrics]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return summary


def configure_cpu_environment(model_n_jobs: Optional[int]) -> None:
    if model_n_jobs is None or int(model_n_jobs) <= 0:
        return
    value = str(int(model_n_jobs))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, value)
