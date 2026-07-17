#!/usr/bin/env python3
"""Build a profile_data.h5 from summit peaks and DNase/H3K27ac BigWigs.

The output schema is compatible with ``ProfileDSWide`` and the production
EPInformer-seq-v2 recipe: ``peak/{chrom,start,summit,profile,counts}``, where
profile has shape ``(N, 2, 1024)`` and channel 0 is DNase cut-site signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyBigWig

WINDOW = 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peaks", required=True, help="narrowPeak BED(.gz) with summit offsets")
    parser.add_argument("--bigwig", required=True, help="DNase 5-prime cut-site BigWig")
    parser.add_argument("--h3k27ac-bigwig", help="optional H3K27ac BigWig for channel 1")
    parser.add_argument("--chrom-sizes", required=True)
    parser.add_argument("--out-h5", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--count-cap-pct", type=float, default=99.9)
    parser.add_argument("--min-count", type=int, default=24)
    args = parser.parse_args()
    chrom_sizes = {
        row.split("\t", 1)[0]: int(row.split("\t", 1)[1].split()[0])
        for row in Path(args.chrom_sizes).read_text().splitlines() if row.strip()
    }
    columns = ["chrom", "start", "end", "name", "score", "strand", "signalValue",
               "pValue", "qValue", "summitOffset"]
    peaks = pd.read_csv(args.peaks, sep="\t", header=None, names=columns,
                        compression="gzip" if args.peaks.endswith(".gz") else None)
    standard = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
    peaks = peaks[peaks.chrom.isin(standard)].copy()
    peaks["summit_abs"] = peaks.start + peaks.summitOffset
    peaks["win_start"] = peaks.summit_abs - WINDOW // 2
    peaks = peaks[(peaks.win_start >= 0) &
                  ((peaks.win_start + WINDOW) <= peaks.chrom.map(chrom_sizes))]
    peaks = peaks.drop_duplicates(["chrom", "win_start"]).reset_index(drop=True)
    dnase = pyBigWig.open(args.bigwig)
    h3 = pyBigWig.open(args.h3k27ac_bigwig) if args.h3k27ac_bigwig else None
    profile = np.zeros((len(peaks), 2, WINDOW), dtype=np.int16)
    counts = np.zeros((len(peaks), 2), dtype=np.int32)
    for i, row in peaks.iterrows():
        values = np.nan_to_num(dnase.values(row.chrom, int(row.win_start),
                                             int(row.win_start) + WINDOW, numpy=True))
        profile[i, 0] = np.clip(np.rint(values), 0, np.iinfo(np.int16).max)
        counts[i, 0] = profile[i, 0].sum()
        if h3 is not None:
            values = np.nan_to_num(h3.values(row.chrom, int(row.win_start),
                                              int(row.win_start) + WINDOW, numpy=True))
            profile[i, 1] = np.clip(np.rint(values), 0, np.iinfo(np.int16).max)
            counts[i, 1] = profile[i, 1].sum()
    dnase.close()
    if h3 is not None:
        h3.close()
    central = profile[:, 0, 12:1012].sum(axis=1)
    cap = np.percentile(central, args.count_cap_pct)
    keep = (central >= args.min_count) & (central <= cap)
    peaks, profile, counts = peaks[keep], profile[keep], counts[keep]
    output = Path(args.out_h5)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        group = handle.create_group("peak")
        group.create_dataset("chrom", data=peaks.chrom.astype("S5"), compression="gzip")
        group.create_dataset("start", data=peaks.win_start.astype(np.int64), compression="gzip")
        group.create_dataset("summit", data=peaks.summit_abs.astype(np.int64), compression="gzip")
        group.create_dataset("profile", data=profile, compression="gzip")
        group.create_dataset("counts", data=counts, compression="gzip")
        group.create_dataset("signal_value", data=peaks.signalValue.astype(np.float32), compression="gzip")
        handle.create_group("bias")
        handle.attrs.update(cell=args.cell, window=WINDOW, n_peaks=len(peaks),
                            source_peaks=args.peaks, source_dnase=args.bigwig,
                            source_h3k27ac=args.h3k27ac_bigwig or "")
    print(f"wrote {len(peaks)} peaks -> {output}")


if __name__ == "__main__":
    main()
