import gc
import hashlib
import os
import re
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress
from transformers import T5EncoderModel, T5Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split

from emulator_bench.common import PROT_MODEL_ID, normalize_smiles


def unikp_sequence_text(sequence: str) -> str:
    sequence = str(sequence)
    if len(sequence) > 1000:
        sequence = sequence[:500] + sequence[-500:]
    sequence = re.sub(r"[UZOB]", "X", sequence)
    return " ".join(list(sequence))


def load_prot_t5(device: torch.device):
    tokenizer = T5Tokenizer.from_pretrained(PROT_MODEL_ID, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROT_MODEL_ID)
    model = model.to(device)
    model.eval()
    gc.collect()
    return tokenizer, model


def protein_cache_file(cache_dir: Path, text: str) -> Path:
    """Cache file of the CataPro wrapper: one fp32 .npy per processed text."""
    key = hashlib.sha256(f"v2|{PROT_MODEL_ID}|max_len=1000|float32|{text}".encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{key}.npy"


def iter_protein_embeddings(
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 8,
) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield (text, vector) per batch for unique processed texts, so callers can save finished work.

    Batches are length-sorted and padded to the longest text, as in the CataPro wrapper.
    The vector is the fp32 mean over residue tokens (the final </s> token is excluded).
    """
    if not texts:
        return
    tokenizer, model = load_prot_t5(device)
    print(f"ProtT5 device: {device} | precision: fp32", flush=True)

    ordered = sorted(texts, key=len, reverse=True)
    for start in progress(range(0, len(ordered), batch_size), desc="ProtT5 embedding", unit="batch"):
        batch = ordered[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(encoded["input_ids"], device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=device)
        with torch.no_grad():
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        for idx, text in enumerate(batch):
            seq_len = int(attention_mask[idx].sum().item())
            yield text, hidden[idx, : seq_len - 1].mean(dim=0).cpu().numpy().astype(np.float32)


def load_smiles_transformer(device: torch.device):
    import __main__

    # The repository vocab.pkl was pickled from a script context, so pickle
    # expects WordVocab to be reachable from __main__ when loaded.
    if not hasattr(__main__, "WordVocab"):
        setattr(__main__, "WordVocab", WordVocab)
    vocab = WordVocab.load_vocab(str(REPO_ROOT / "vocab.pkl"))
    model = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    state = torch.load(str(REPO_ROOT / "trfm_12_23000.pkl"), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return vocab, model


def smiles_to_ids(smiles: str, vocab) -> List[int]:
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    tokens = split(normalize_smiles(smiles)).split()
    if len(tokens) > 218:
        tokens = tokens[:109] + tokens[-109:]
    ids = [sos_index] + [vocab.stoi.get(token, unk_index) for token in tokens] + [eos_index]
    ids.extend([pad_index] * (220 - len(ids)))
    return ids


def _encode_smiles_batch(model, src: torch.Tensor, columns: torch.Tensor) -> np.ndarray:
    """Encode src (T, B). Column b gets pe[b], as in the original, where the pe slice runs over the batch axis."""
    with torch.no_grad():
        embedded = model.embed(src)
        embedded = embedded + model.pe.pe[0, columns]
        output = embedded
        for idx in range(model.trfm.encoder.num_layers - 1):
            output = model.trfm.encoder.layers[idx](output, src_mask=None)
        penul = output
        output = model.trfm.encoder.layers[-1](output, src_mask=None)
        if model.trfm.encoder.norm:
            output = model.trfm.encoder.norm(output)
        output_f = output.detach().float().cpu().numpy()
        penul_f = penul.detach().float().cpu().numpy()
    return np.hstack(
        [
            np.mean(output_f, axis=0),
            np.max(output_f, axis=0),
            output_f[0, :, :],
            penul_f[0, :, :],
        ]
    ).astype(np.float32, copy=False)


def iter_smiles_embeddings(
    pairs: Sequence[Tuple[str, int]],
    device: torch.device,
    batch_size: int = 256,
) -> Iterator[Tuple[Tuple[str, int], np.ndarray]]:
    """Yield ((smiles, column), vector) after each batch so callers can save finished work.

    Inputs are padded to 220 tokens, so a vector depends only on the SMILES and its column.
    """
    if not pairs:
        return
    vocab, model = load_smiles_transformer(device)
    print(f"SMILES Transformer device: {device} | precision: fp32", flush=True)

    unique = sorted({(normalize_smiles(smiles), int(column)) for smiles, column in pairs}, key=lambda pair: len(pair[0]), reverse=True)
    for start in progress(range(0, len(unique), batch_size), desc="SMILES Transformer embedding", unit="batch"):
        batch = unique[start : start + batch_size]
        ids = [smiles_to_ids(smiles, vocab) for smiles, _ in batch]
        src = torch.tensor(ids, dtype=torch.long, device=device).t().contiguous()
        columns = torch.tensor([column for _, column in batch], dtype=torch.long, device=device)
        vectors = _encode_smiles_batch(model, src, columns)
        for idx, pair in enumerate(batch):
            yield pair, vectors[idx]


def save_npz_atomic(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "wb") as handle:
        np.savez(handle, **payload)
    tmp_path.replace(path)


def save_npy_atomic(path: Path, vector: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}.npy")
    np.save(tmp_path, vector.astype(np.float32, copy=False))
    tmp_path.replace(path)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, zlib.error) as exc:
        raise RuntimeError(f"Corrupted cache file: {path}. Rebuild it with --overwrite.") from exc


def load_cached_vector(path: Path, key: str = "vector") -> np.ndarray:
    payload = load_npz(path)
    if key not in payload:
        raise KeyError(f"Missing `{key}` in cache file {path}")
    return payload[key].astype(np.float32, copy=False)
