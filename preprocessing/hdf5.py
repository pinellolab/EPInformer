"""
Compressed HDF5 output for promoter–enhancer preprocessing (one file per run).

Factored schema — enhancer sequences stored once, referenced by index:

Gene-level datasets (row index ``i`` over N genes):

- ``ensid`` (str, shape (N,)) — gene / sample IDs
- ``promoter_seq`` (float32, shape (N, L, 4)) — promoter one-hot sequences
- ``gene_enh_idx`` (int32, shape (N, n_enh)) — index into enhancer_seq; -1 = padding
- ``activity`` (float32, shape (N, n_enh)) — per-enhancer activity_base
- ``dhs`` (float32, shape (N, n_enh)) — normalized DHS signal
- ``distance`` (float32, shape (N, n_enh)) — distance_relative to TSS
- ``contact`` (float32, shape (N, n_enh)) — Hi-C contact

Enhancer-level datasets (row index ``j`` over M unique enhancers):

- ``enhancer_seq`` (float32, shape (M, L, 4)) — enhancer one-hot sequences
- ``enhancer_name`` (str, shape (M,)) — enhancer element IDs

Attributes: ``max_n_enhancer``, ``max_seq_len``, ``n_signal_tracks``, ``n_enhancers``.
"""

from __future__ import annotations

from typing import Any, Optional

import h5py
import numpy as np


def _gzip_opts(level: int = 4) -> dict:
    return {"compression": "gzip", "compression_opts": level}


def create_pe_arrays_h5(
    path: str,
    n_samples: int,
    n_enhancers: int,
    max_n_enhancer: int,
    max_seq_len: int,
    n_signal_tracks: int = 0,
    compression: str = "gzip",
    compression_opts: int = 4,
    chunk_samples: int = 1,
) -> h5py.File:
    """
    Create an HDF5 file with the factored schema.

    Parameters
    ----------
    n_samples : int
        Number of genes (N).
    n_enhancers : int
        Number of unique enhancers (M).
    max_n_enhancer : int
        Max enhancers per gene (padding width).
    max_seq_len : int
        Sequence length (e.g. 2000).
    n_signal_tracks : int
        Number of BigWig signal tracks (0 = omit signal datasets).
    """
    kw = {}
    if compression == "gzip":
        kw = _gzip_opts(compression_opts)
    elif compression == "lzf":
        kw = {"compression": "lzf"}

    f = h5py.File(path, "w")
    f.attrs["max_n_enhancer"] = max_n_enhancer
    f.attrs["max_seq_len"] = max_seq_len
    f.attrs["n_signal_tracks"] = n_signal_tracks
    f.attrs["n_enhancers"] = n_enhancers

    cs = min(chunk_samples, n_samples)

    # Gene-level: promoter sequences
    f.create_dataset(
        "promoter_seq",
        shape=(n_samples, max_seq_len, 4),
        dtype=np.float32,
        chunks=(cs, max_seq_len, 4),
        **kw,
    )

    # Gene-level: enhancer index matrix (-1 = no enhancer)
    f.create_dataset(
        "gene_enh_idx",
        shape=(n_samples, max_n_enhancer),
        dtype=np.int32,
        chunks=(cs, max_n_enhancer),
        fillvalue=-1,
        **kw,
    )

    # Gene-enhancer pair features
    fe_kw = {**kw, "chunks": (cs, max_n_enhancer)}
    for name in ("activity", "dhs", "distance", "contact"):
        f.create_dataset(
            name,
            shape=(n_samples, max_n_enhancer),
            dtype=np.float32,
            **fe_kw,
        )

    # Gene IDs
    dt = h5py.string_dtype(encoding="utf-8", length=None)
    f.create_dataset("ensid", shape=(n_samples,), dtype=dt)

    # Enhancer-level: sequences (stored once per unique enhancer)
    ce = min(chunk_samples, max(1, n_enhancers))
    f.create_dataset(
        "enhancer_seq",
        shape=(n_enhancers, max_seq_len, 4),
        dtype=np.float32,
        chunks=(ce, max_seq_len, 4),
        **kw,
    )

    # Enhancer-level: names
    f.create_dataset("enhancer_name", shape=(n_enhancers,), dtype=dt)

    # Optional signal tracks
    if n_signal_tracks > 0:
        f.create_dataset(
            "promoter_signal",
            shape=(n_samples, max_seq_len, n_signal_tracks),
            dtype=np.float32,
            chunks=(cs, max_seq_len, n_signal_tracks),
            **kw,
        )
        f.create_dataset(
            "enhancer_signal",
            shape=(n_enhancers, max_seq_len, n_signal_tracks),
            dtype=np.float32,
            chunks=(ce, max_seq_len, n_signal_tracks),
            **kw,
        )

    return f


def write_enhancer(
    f: h5py.File,
    enh_index: int,
    name: str,
    seq: np.ndarray,
    signal: Optional[np.ndarray] = None,
) -> None:
    """Write a single unique enhancer into row ``enh_index``."""
    f["enhancer_name"][enh_index] = name
    f["enhancer_seq"][enh_index] = seq.astype(np.float32, copy=False)
    if signal is not None and "enhancer_signal" in f:
        f["enhancer_signal"][enh_index] = signal.astype(np.float32, copy=False)


def write_gene_sample(
    f: h5py.File,
    gene_index: int,
    ensid: str,
    promoter_seq: np.ndarray,
    enh_indices: np.ndarray,
    activity: np.ndarray,
    dhs: np.ndarray,
    distance: np.ndarray,
    contact: np.ndarray,
    promoter_signal: Optional[np.ndarray] = None,
) -> None:
    """Write one gene sample into row ``gene_index``."""
    f["ensid"][gene_index] = ensid
    f["promoter_seq"][gene_index] = promoter_seq.astype(np.float32, copy=False)
    f["gene_enh_idx"][gene_index] = enh_indices.astype(np.int32, copy=False)
    f["activity"][gene_index] = activity.astype(np.float32, copy=False)
    f["dhs"][gene_index] = dhs.astype(np.float32, copy=False)
    f["distance"][gene_index] = distance.astype(np.float32, copy=False)
    f["contact"][gene_index] = contact.astype(np.float32, copy=False)
    if promoter_signal is not None and "promoter_signal" in f:
        f["promoter_signal"][gene_index] = promoter_signal.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Legacy compatibility wrappers (used by old pipelines that pass seq_code)
# ---------------------------------------------------------------------------

def normalize_seq_signal(
    seq_signal: np.ndarray, n_signal_tracks: int
) -> np.ndarray:
    """Broadcast single-track (n_tok, L) to (n_tok, L, 1) when needed."""
    x = np.asarray(seq_signal, dtype=np.float32)
    if n_signal_tracks <= 0:
        return x
    if x.ndim == 2:
        return x[..., np.newaxis]
    return x


def pack_obtain_pe_payload(
    seq_code,
    act_list,
    dhs_list,
    dist_list,
    contact_list,
    seq_signal,
) -> tuple[np.ndarray, ...]:
    """
    Convert legacy components into arrays suitable for write functions.
    """
    return (
        np.asarray(seq_code, dtype=np.float32),
        np.asarray(act_list, dtype=np.float32),
        np.asarray(dhs_list, dtype=np.float32),
        np.asarray(dist_list, dtype=np.float32),
        np.asarray(contact_list, dtype=np.float32),
        np.asarray(seq_signal, dtype=np.float32) if seq_signal is not None else None,
    )


def read_pe_h5(path: str) -> dict[str, Any]:
    """Load all datasets into memory (small/medium files)."""
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = f[k][:]
        out["attrs"] = dict(f.attrs)
    return out
