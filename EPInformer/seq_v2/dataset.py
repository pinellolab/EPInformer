"""HDF5 + FASTA dataset for 2,114-bp EPInformer-seq-v2 training windows."""

from __future__ import annotations

import random
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_BASE_IDX = np.full(256, 4, dtype=np.uint8)
for _base, _index in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _BASE_IDX[ord(_base)] = _index
    _BASE_IDX[ord(_base.lower())] = _index
_LUT = np.eye(5, dtype=np.float32)[:, :4].T.copy()


class ProfileDSWide(Dataset):
    """Cache one cell's HDF5 profiles and extract matching FASTA windows."""

    def __init__(self, h5_path: str, chroms_keep: Iterable[str], fasta_path: str,
                 group: str = "peak", reverse_complement: bool = True,
                 in_window: int = 2114, out_window: int = 1024):
        if in_window < out_window:
            raise ValueError("in_window must be at least out_window")
        self.rc = reverse_complement
        self.in_window, self.out_window = in_window, out_window
        pad_left = (in_window - out_window) // 2
        with h5py.File(h5_path, "r") as handle:
            group_data = handle[group]
            chrom = group_data["chrom"][:].astype(str)
            keep = np.where(np.isin(chrom, list(chroms_keep)))[0]
            self.profile = group_data["profile"][keep].astype(np.int16)
            self.counts = group_data["counts"][keep].astype(np.int32)
            starts = group_data["start"][keep].astype(np.int64)
            chrom = chrom[keep]
        import pyfaidx
        fasta = pyfaidx.Fasta(fasta_path, sequence_always_upper=True)
        lengths = {name: len(fasta[name]) for name in set(chrom.tolist())}
        self.seq_idx = np.empty((len(keep), in_window), dtype=np.uint8)
        for row, (name, start) in enumerate(zip(chrom, starts)):
            begin = int(start) - pad_left
            end = begin + in_window
            left = max(0, -begin)
            right = max(0, end - lengths[name])
            seq = str(fasta[name][max(0, begin):min(lengths[name], end)].seq)
            seq = ("N" * left) + seq + ("N" * right)
            self.seq_idx[row] = _BASE_IDX[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
        fasta.close()

    def __len__(self) -> int:
        return len(self.seq_idx)

    def __getitem__(self, index: int):
        one_hot = _LUT[:, self.seq_idx[index]]
        profile = self.profile[index].astype(np.float32)
        counts = self.counts[index].astype(np.float32)
        if self.rc and random.random() > 0.5:
            one_hot = one_hot[::-1, ::-1].copy()
            profile = profile[:, ::-1].copy()
        return torch.from_numpy(one_hot), torch.from_numpy(profile), torch.from_numpy(counts)
