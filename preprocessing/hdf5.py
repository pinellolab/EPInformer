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
    n_enhancers: Optional[int] = None,
    max_n_enhancer: Optional[int] = None,
    max_seq_len: Optional[int] = None,
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
    if max_n_enhancer is None or max_seq_len is None:
        raise TypeError("max_n_enhancer and max_seq_len are required")
    if n_enhancers is None:
        # Legacy preprocessing passed one promoter+enhancer tensor per gene and
        # had no global enhancer-name table. Reserve deterministic per-gene
        # slots; write_pe_sample only materializes the slots that are populated.
        n_enhancers = n_samples * max_n_enhancer

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


def write_pe_sample(
    f: h5py.File,
    gene_index: int,
    ensid: str,
    seq_code,
    activity,
    dhs,
    distance,
    contact,
    seq_signal=None,
    n_signal_tracks: int = 0,
) -> None:
    """Write a legacy promoter+enhancer tensor into the factored schema.

    Legacy extractors return ``seq_code`` with the promoter in slot 0 and up to
    ``max_n_enhancer`` enhancer sequences after it, but do not return a stable
    global enhancer-name mapping. Each populated enhancer therefore receives a
    deterministic per-gene index and synthetic name. Supported modern builders
    should continue to use :func:`write_enhancer` and :func:`write_gene_sample`
    so shared enhancers are stored only once.
    """
    seq = np.asarray(seq_code, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[-1] != 4 or seq.shape[0] < 1:
        raise ValueError(
            "legacy seq_code must have shape (1+n_enhancers, sequence_length, 4)"
        )

    max_n_enhancer = int(f.attrs["max_n_enhancer"])
    max_seq_len = int(f.attrs["max_seq_len"])
    if seq.shape[1:] != (max_seq_len, 4):
        raise ValueError(
            f"legacy seq_code has sequence shape {seq.shape[1:]}; "
            f"expected ({max_seq_len}, 4)"
        )

    def _padded(values) -> np.ndarray:
        out = np.zeros(max_n_enhancer, dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        out[: min(len(arr), max_n_enhancer)] = arr[:max_n_enhancer]
        return out

    act_arr = _padded(activity)
    dhs_arr = _padded(dhs)
    dist_arr = _padded(distance)
    contact_arr = _padded(contact)
    enh_indices = np.full(max_n_enhancer, -1, dtype=np.int32)

    signals = None
    if seq_signal is not None and n_signal_tracks > 0:
        signals = normalize_seq_signal(seq_signal, n_signal_tracks)
        if signals.ndim != 3 or signals.shape[1:] != (max_seq_len, n_signal_tracks):
            raise ValueError(
                "legacy seq_signal must have shape "
                f"(1+n_enhancers, {max_seq_len}, {n_signal_tracks})"
            )

    n_slots = min(max_n_enhancer, seq.shape[0] - 1)
    for slot in range(n_slots):
        enh_seq = seq[slot + 1]
        populated = bool(
            np.any(enh_seq)
            or act_arr[slot] != 0
            or dhs_arr[slot] != 0
            or dist_arr[slot] != 0
            or contact_arr[slot] != 0
        )
        if not populated:
            continue
        enh_index = gene_index * max_n_enhancer + slot
        if enh_index >= f["enhancer_seq"].shape[0]:
            raise IndexError("legacy enhancer slot exceeds reserved HDF5 capacity")
        enh_signal = signals[slot + 1] if signals is not None else None
        write_enhancer(
            f,
            enh_index,
            f"{ensid}:legacy_slot_{slot}",
            enh_seq,
            signal=enh_signal,
        )
        enh_indices[slot] = enh_index

    promoter_signal = signals[0] if signals is not None else None
    write_gene_sample(
        f,
        gene_index,
        ensid,
        seq[0],
        enh_indices,
        act_arr,
        dhs_arr,
        dist_arr,
        contact_arr,
        promoter_signal=promoter_signal,
    )


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
