"""Equivalence of the emulator_bench wrapper with the original UniKP code.

Run from baselines/UniKP (see emulator_bench/verify_equivalence.sh):
    CUDA_VISIBLE_DEVICES= python -m pytest tests -m 'not gpu'      # CPU, tiny random T5, committed fixture
    python -m pytest tests -m gpu -s                                # real ProtT5 on one GPU, EMULaToR data
"""

import __main__
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(ROOT))

import UniKP_kcat as og  # noqa: E402  original smiles_to_vec and Seq_to_vec
from build_vocab import WordVocab  # noqa: E402
from emulator_bench import build_split_features, cache_embeddings, feature_pipeline as fp  # noqa: E402
from emulator_bench import train_single_target_tvt as train  # noqa: E402
from emulator_bench.common import DEFAULT_PROTEIN_CACHE_DIR, TaskTable  # noqa: E402

# The original scripts run as __main__, where the pickled vocab finds WordVocab.
__main__.WordVocab = WordVocab

# Batched, masked and fp32-mean ProtT5 against the original one-at-a-time float64 mean.
PROT_RTOL, PROT_ATOL = 1e-4, 1e-5


class OriginalTorch:
    """The `torch` the original script sees. Two environment shims: the checkpoint was saved on cuda:0, so it
    loads to the CPU; the script uses cuda:1, which is cuda:0 on a one-GPU machine."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def load(path, **kwargs):
        return torch.load(path, map_location="cpu", **kwargs)

    @staticmethod
    def device(name):
        return torch.device("cuda:0" if name == "cuda:1" else name)


class RecordingTokenizer:
    """Wraps a tokenizer and records the texts the original code passes to batch_encode_plus."""

    def __init__(self, tokenizer):
        self.tokenizer, self.texts = tokenizer, []

    def batch_encode_plus(self, texts, **kwargs):
        self.texts.extend(texts)
        return self.tokenizer(texts, **kwargs)  # transformers 5 removed batch_encode_plus

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


@pytest.fixture(scope="module")
def tiny_t5():
    """Random tiny T5 encoder and a committed tiny sentencepiece tokenizer, shared by original and wrapper."""
    from transformers import T5Config, T5EncoderModel, T5Tokenizer

    torch.manual_seed(0)
    model = T5EncoderModel(T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4)).eval()
    tokenizer = RecordingTokenizer(T5Tokenizer.from_pretrained(FIXTURES / "tiny_t5_tokenizer", extra_ids=0))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(T5EncoderModel, "from_pretrained", lambda *a, **kw: model)
        mp.setattr(T5Tokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
        yield tokenizer


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(FIXTURES / "tiny_task_table.csv", keep_default_na=False)


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)  # the original code opens vocab.pkl and the weights by relative path
    monkeypatch.setattr(og, "torch", OriginalTorch())


def test_fixture_covers_cases(table):
    assert len(table) >= 150
    assert table["smiles"].str.contains(".", regex=False).sum() >= 10
    assert table["smiles"].map(lambda s: len(og.split(s).split())).max() > 218
    assert table["sequence"].str.contains("[UZOB]").sum() >= 10


def test_smiles_vectors_equal_original(table):
    """Rows cross the 100-row chunk boundary; each wrapper vector uses its table column (row % 100)."""
    expected = og.smiles_to_vec(list(table["smiles"]))
    columns = np.arange(len(table)) % 100
    pairs = list(zip(table["smiles"], columns))
    vectors = dict(fp.iter_smiles_embeddings(pairs, torch.device("cpu"), batch_size=512))
    got = np.stack([vectors[(smiles, column)] for smiles, column in pairs])
    assert got.shape == (len(table), 1024) and len(table) > 100
    np.testing.assert_array_equal(got, expected.astype(np.float32))


def test_task_table_columns_and_missing_rows(table, tmp_path):
    root = tmp_path / "kcat"
    root.mkdir()
    table.to_parquet(root / "kcat_kinetic_params_1d.parquet")
    task_table = TaskTable(tmp_path, "kcat")
    shuffled = table.sample(frac=1, random_state=1)
    assert (task_table.attach_columns(shuffled)["trfm_column"].to_numpy() == shuffled.index.to_numpy() % 100).all()
    with pytest.raises(ValueError, match="not in the task table"):
        task_table.attach_columns(table.assign(value=table["value"] + 1))


def test_protein_text_equals_original(tiny_t5, table):
    long = "MKTAYIAKQR" * 130
    sequences = [long, long[::-1] + "UZOB", "MKTAYUZOBQ", *table["sequence"][:6]]
    tiny_t5.texts.clear()
    og.Seq_to_vec(list(sequences))
    assert [fp.unikp_sequence_text(s) for s in sequences] == tiny_t5.texts
    assert tiny_t5.texts[0].count(" ") == 999 and "X" in tiny_t5.texts[1]


def test_protein_vectors_equal_original(tiny_t5, table):
    long = "MKTAYIAKQRQISFVKSH" * 70
    sequences = [long, "MKTAYIAKQR", "UZOBMKT", *table["sequence"][:20]]
    expected = og.Seq_to_vec(list(sequences))
    texts = sorted({fp.unikp_sequence_text(s) for s in sequences})
    vectors = dict(fp.iter_protein_embeddings(texts, torch.device("cpu"), batch_size=3))
    got = np.stack([vectors[fp.unikp_sequence_text(s)] for s in sequences])
    np.testing.assert_allclose(got, expected, rtol=PROT_RTOL, atol=PROT_ATOL)


@pytest.mark.parametrize("drop_dot", [True, False], ids=["kcat_drop_dot_train", "km_keep_all"])
def test_feature_assembly_equals_original(tiny_t5, table, tmp_path, drop_dot):
    """Columns are SMILES (1024) then protein (1024). Dots are dropped from train only, and only for kcat."""
    root, work = tmp_path / "data", tmp_path / "work"
    split_dir = root / "kcat" / "random_splits_grouped_sequence"
    split_dir.mkdir(parents=True)
    table.to_parquet(root / "kcat" / "kcat_kinetic_params_1d.parquet")
    order = np.random.default_rng(0).permutation(len(table))
    parts = {"train": order[:120], "val": order[120:145], "test": order[145:]}
    for name, rows in parts.items():
        table.iloc[rows].to_parquet(split_dir / f"{name}.parquet")

    feature = np.concatenate([og.smiles_to_vec(list(table["smiles"])), og.Seq_to_vec(list(table["sequence"]))], axis=1)
    flags = ["--tasks", "kcat", "--dataset_root", str(root), "--work_dir", str(work), "--protein_cache_dir", str(tmp_path / "prot"), *(["--drop_dot_smiles_train"] * drop_dot)]
    for module, extra in ((cache_embeddings, ["--device", "cpu"]), (build_split_features, [])):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["x", *flags, *extra])
            module.main()
        with pytest.MonkeyPatch.context() as mp:  # second run is a no-op
            mp.setattr(sys, "argv", ["x", *flags, *extra])
            module.main()

    for name, rows in parts.items():
        if name == "train" and drop_dot:
            rows = rows[~table["smiles"].iloc[rows].str.contains(".", regex=False).to_numpy()]
        with np.load(work / "kcat" / "feature_matrices" / "random_splits_grouped_sequence" / "all" / f"{name}.npz") as data:
            x, y = data["X"], data["y"]
        assert x.shape == (len(rows), feature.shape[1])  # 1024 SMILES + the tiny T5 width
        np.testing.assert_array_equal(x[:, :1024], feature[rows, :1024].astype(np.float32))
        np.testing.assert_allclose(x[:, 1024:], feature[rows, 1024:], rtol=PROT_RTOL, atol=PROT_ATOL)
        np.testing.assert_array_equal(y, table["log10_value"].to_numpy()[rows])
    assert table["smiles"].iloc[parts["test"]].str.contains(".", regex=False).any()


def test_extra_trees_parameters_and_seed(tmp_path):
    from sklearn.ensemble import ExtraTreesRegressor

    for name in ("UniKP_kcat.py", "UniKP_Km.py", "UniKP_kcat_model.py"):
        assert "ExtraTreesRegressor()" in (ROOT / name).read_text()
    original = ExtraTreesRegressor().get_params()
    params = train.make_model(666, 4).get_params()
    assert {k: v for k, v in params.items() if k in ("n_jobs", "random_state")} == {"n_jobs": 4, "random_state": 666}
    assert {k: v for k, v in params.items() if k not in ("n_jobs", "random_state")} == {k: v for k, v in original.items() if k not in ("n_jobs", "random_state")}

    rng = np.random.default_rng(0)
    x_train, x_test = rng.normal(size=(300, 2048)).astype(np.float32), rng.normal(size=(40, 2048)).astype(np.float32)
    y_train = x_train[:, 0] + rng.normal(size=300)
    direct = ExtraTreesRegressor(random_state=666).fit(x_train, y_train).predict(x_test)
    one = train.make_model(666, 1).fit(x_train, y_train)
    eight = train.make_model(666, 8).fit(x_train, y_train)
    reference = ExtraTreesRegressor(random_state=666).fit(x_train, y_train)
    for model in (one, eight):  # n_jobs does not change any tree
        for tree, expected in zip(model.estimators_, reference.estimators_):
            for field in ("feature", "threshold", "value", "children_left", "children_right"):
                np.testing.assert_array_equal(getattr(tree.tree_, field), getattr(expected.tree_, field))
        # Parallel predict adds the tree outputs in thread order, so the sum differs by float64 rounding.
        np.testing.assert_allclose(model.predict(x_test), direct, rtol=0, atol=1e-12)
    assert not np.allclose(train.make_model(777, 1).fit(x_train, y_train).predict(x_test), direct)


def test_training_script_equals_direct_fit(tmp_path):
    from sklearn.ensemble import ExtraTreesRegressor

    rng = np.random.default_rng(1)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    data = {}
    for split, n in (("train", 200), ("val", 30), ("test", 30)):
        x, y = rng.normal(size=(n, 2048)).astype(np.float32), rng.normal(size=n)
        np.savez(feature_dir / f"{split}.npz", X=x, y=y)
        pd.DataFrame({"sequence": ["A"] * n}).to_csv(feature_dir / f"{split}_metadata.csv", index=False)
        data[split] = (x, y)
    for n_jobs in (1, 8):
        out = tmp_path / f"run_{n_jobs}"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["x", "--feature_dir", str(feature_dir), "--out_dir", str(out), "--seed", "666", "--model_n_jobs", str(n_jobs)])
            train.main()
        direct = ExtraTreesRegressor(random_state=666).fit(*data["train"]).predict(data["test"][0])
        saved = pd.read_csv(out / "predictions_test.csv")["y_pred"].to_numpy()
        np.testing.assert_allclose(saved, direct, rtol=0, atol=1e-12)


@pytest.mark.gpu
def test_smiles_cache_equals_original_on_real_rows():
    """The rebuilt (CPU) kcat SMILES cache equals the original smiles_to_vec on the first 250 real table rows, exactly."""
    from emulator_bench.common import DEFAULT_DATASET_ROOT, DEFAULT_WORK_DIR, smiles_cache_path

    smiles = pd.read_parquet(DEFAULT_DATASET_ROOT / "kcat/kcat_kinetic_params_1d.parquet", columns=["smiles"])["smiles"][:250]
    expected = og.smiles_to_vec(list(smiles)).astype(np.float32)
    fingerprint = TaskTable(DEFAULT_DATASET_ROOT, "kcat").fingerprint
    got = np.stack([fp.load_cached_vector(smiles_cache_path(DEFAULT_WORK_DIR / "embeddings", s, i % 100, fingerprint)) for i, s in enumerate(smiles)])
    print(f"\nSMILES cache vs original: 250 rows, max abs diff {np.abs(got - expected).max():.3e}")
    np.testing.assert_array_equal(got, expected)


@pytest.mark.gpu
def test_real_prott5_cache_equals_original_one_at_a_time():
    """Original one-at-a-time fp32 ProtT5 on a length-stratified sample against the CataPro cache vectors."""
    from transformers import T5EncoderModel, T5Tokenizer

    assert torch.cuda.is_available(), "run with a visible GPU"
    table = pd.read_parquet(Path(__file__).resolve().parents[3] / "data/processed/datasets/kinetic_dataset/kcat/kcat_kinetic_params_1d.parquet", columns=["sequence"])
    unique = sorted(table["sequence"].unique(), key=len)
    picks = {unique[i] for i in np.linspace(0, len(unique) - 1, 24).astype(int)}
    picks.update(unique[-6:])  # the longest sequences
    picks.update([s for s in unique if len(s) > 1000][::40][:6])
    sequences = sorted(picks, key=len)
    assert sum(len(s) > 1000 for s in sequences) >= 6

    # The original script loads the weights from a local folder; use the Hub weights from the local cache.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(T5Tokenizer, "batch_encode_plus", lambda self, texts, **kw: self(texts, **kw), raising=False)
        for cls in (T5EncoderModel, T5Tokenizer):
            mp.setattr(cls, "from_pretrained", lambda name, _orig=cls.from_pretrained, **kw: _orig(fp.PROT_MODEL_ID, local_files_only=True, **kw))
        expected = og.Seq_to_vec(list(sequences))

    cached = np.stack([np.load(fp.protein_cache_file(DEFAULT_PROTEIN_CACHE_DIR, fp.unikp_sequence_text(s))) for s in sequences])
    abs_diff = np.abs(cached - expected)
    rel_diff = abs_diff / np.maximum(np.abs(expected), 1e-3)
    print(f"\nProtT5 cache vs original one-at-a-time: {len(sequences)} sequences, lengths {len(sequences[0])}..{len(sequences[-1])}")
    print(f"max abs diff {abs_diff.max():.3e}, max rel diff {rel_diff.max():.3e} (|x| floor 1e-3), mean abs diff {abs_diff.mean():.3e}")
    for s, a in zip(sequences, abs_diff.max(axis=1)):
        print(f"  len {len(s):5d} max abs diff {a:.3e}")
    np.testing.assert_allclose(cached, expected, rtol=1e-3, atol=1e-5)

