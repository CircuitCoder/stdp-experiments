from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import MODEL


PORTABLE_CHECKPOINT_SCHEMA = "zd3-portable-checkpoint-v1"


@dataclass(frozen=True)
class MnistSplit:
    images: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class PortableCheckpoint:
    weights: np.ndarray
    theta_mv: np.ndarray
    accepted_samples: int
    manifest: dict[str, Any]


def _read_idx(path: Path, expected_dims: int) -> np.ndarray:
    with path.open("rb") as stream:
        magic = stream.read(4)
        if len(magic) != 4 or magic[:2] != b"\x00\x00":
            raise ValueError(f"{path} is not an IDX file")
        dtype_code = magic[2]
        dims = magic[3]
        if dtype_code != 0x08 or dims != expected_dims:
            raise ValueError(
                f"{path} has IDX dtype/dims {dtype_code:#x}/{dims}, expected 0x08/{expected_dims}"
            )
        shape = struct.unpack(">" + "I" * dims, stream.read(4 * dims))
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path} contains {data.size} values, expected {expected}")
    return data.reshape(shape).copy()


def load_mnist(data_dir: Path, split: str) -> MnistSplit:
    names = {
        "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
        "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
    }
    if split not in names:
        raise ValueError(f"unknown MNIST split: {split}")
    image_name, label_name = names[split]
    images = _read_idx(data_dir / image_name, 3).reshape(-1, MODEL.n_input)
    labels = _read_idx(data_dir / label_name, 1)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("MNIST image and label counts differ")
    return MnistSplit(images=images, labels=labels)


def load_reference_triplets(
    path: Path, rows: int, columns: int, *, dtype: np.dtype = np.float64
) -> np.ndarray:
    triplets = np.load(path, allow_pickle=False)
    if triplets.ndim != 2 or triplets.shape[1] != 3:
        raise ValueError(f"{path} must contain [pre, post, weight] triplets")
    pre = triplets[:, 0].astype(np.int64)
    post = triplets[:, 1].astype(np.int64)
    if np.any(pre < 0) or np.any(pre >= rows) or np.any(post < 0) or np.any(post >= columns):
        raise ValueError(f"{path} contains an out-of-range connection index")
    matrix = np.zeros((rows, columns), dtype=dtype)
    matrix[pre, post] = triplets[:, 2].astype(dtype, copy=False)
    return matrix


def normalize_columns(weights: np.ndarray, target: float = MODEL.normalization_target) -> None:
    sums = weights.sum(axis=0, dtype=np.float64)
    if np.any(sums <= 0.0) or not np.all(np.isfinite(sums)):
        raise ValueError("cannot normalize a non-finite or zero-sum weight column")
    weights *= (target / sums).astype(weights.dtype, copy=False)[None, :]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    *,
    weights: np.ndarray,
    theta_mv: np.ndarray,
    accepted_samples: int,
    manifest: dict[str, Any],
) -> None:
    if weights.shape != (MODEL.n_input, MODEL.n_exc):
        raise ValueError(f"unexpected feedforward shape {weights.shape}")
    if theta_mv.shape != (MODEL.n_exc,):
        raise ValueError(f"unexpected theta shape {theta_mv.shape}")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(theta_mv)):
        raise ValueError("checkpoint contains non-finite model state")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    complete_manifest = {
        "checkpoint_schema": PORTABLE_CHECKPOINT_SCHEMA,
        "model": MODEL.as_dict(),
        "matrix_layout": "weights[input, excitatory], C row-major",
        **manifest,
    }
    manifest_json = json.dumps(
        complete_manifest, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            np.savez(
                stream,
                weights=np.asarray(weights, dtype=np.float64, order="C"),
                theta_mv=np.asarray(theta_mv, dtype=np.float64),
                accepted_samples=np.asarray(accepted_samples, dtype=np.int64),
                manifest_json=np.frombuffer(manifest_json, dtype=np.uint8),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path) -> PortableCheckpoint:
    with np.load(path, allow_pickle=False) as archive:
        required = {"weights", "theta_mv", "accepted_samples", "manifest_json"}
        if set(archive.files) != required:
            raise ValueError(f"{path} has unexpected checkpoint fields: {archive.files}")
        weights = np.asarray(archive["weights"], dtype=np.float64, order="C")
        theta_mv = np.asarray(archive["theta_mv"], dtype=np.float64)
        accepted_samples = int(archive["accepted_samples"])
        manifest = json.loads(bytes(archive["manifest_json"]).decode("ascii"))
    if manifest.get("checkpoint_schema") != PORTABLE_CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema in {path}")
    if weights.shape != (MODEL.n_input, MODEL.n_exc):
        raise ValueError(f"unexpected feedforward shape {weights.shape}")
    if theta_mv.shape != (MODEL.n_exc,):
        raise ValueError(f"unexpected theta shape {theta_mv.shape}")
    return PortableCheckpoint(weights, theta_mv, accepted_samples, manifest)
