#!/usr/bin/env python3
"""Read-only stats: legacy Pinello-style HDF5 vs factored EPInformer HDF5 (NaNs + ranges)."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _stats(name: str, x: np.ndarray) -> None:
    x = np.asarray(x)
    n = x.size
    n_nan = int(np.isnan(x).sum())
    finite = x[np.isfinite(x)]
    print(f"  {name}: shape={x.shape}, dtype={x.dtype}")
    print(f"    nan_count={n_nan} ({100.0 * n_nan / max(n, 1):.4f}% of elements)")
    if finite.size:
        print(
            f"    finite: min={finite.min():.6g}, max={finite.max():.6g}, mean={finite.mean():.6g}"
        )
    else:
        print("    finite: (none)")


def legacy_report(path: str) -> None:
    import h5py

    with h5py.File(path, "r") as f:
        print(f"Legacy HDF5: {path}")
        print("  keys:", list(f.keys()))
        ef = np.asarray(f["enhancers_feat"][:], dtype=np.float64)
    # Columns: 0 distance, 1 DHS.RPM, 2 H3K27ac.RPM, 3 activity_base_no_qnorm, 4 hic_contact
    labels = [
        "col0_distance",
        "col1_DHS_RPM",
        "col2_H3K27ac_RPM",
        "col3_activity_base_no_qnorm",
        "col4_hic_contact",
    ]
    for i, lab in enumerate(labels):
        _stats(lab, ef[..., i])
    # Same 3 channels as training: abs(d), col3, col4
    d, a, c = np.abs(ef[..., 0]), ef[..., 3], ef[..., 4]
    _stats("model_path_abs_distance", d)
    _stats("model_path_activity_col3", a)
    _stats("model_path_contact_col4", c)


def factored_report(path: str) -> None:
    import h5py

    with h5py.File(path, "r") as f:
        print(f"Factored HDF5: {path}")
        print("  keys:", list(f.keys()))
        activity = np.asarray(f["activity"][:], dtype=np.float64)
        contact = np.asarray(f["contact"][:], dtype=np.float64)
        distance = np.asarray(f["distance"][:], dtype=np.float64)

    _stats("activity", activity)
    _stats("contact", contact)
    _stats("distance", distance)

    # Optional mask via gene_enh_idx
    with h5py.File(path, "r") as f:
        if "gene_enh_idx" not in f:
            return
        idx = f["gene_enh_idx"][:]
    mask = idx >= 0
    print(f"  gene_enh_idx padding: {(~mask).sum()} padded / {mask.size} total slots")
    if mask.any():
        _stats("activity (slots with gene_enh_idx>=0)", activity[mask])
        _stats("contact (slots with gene_enh_idx>=0)", contact[mask])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-h5", type=str, default=None, help="Legacy *200CREs*gene_RPM_4feats.hdf5")
    p.add_argument("--factored-h5", type=str, default=None, help="Factored samples.h5")
    args = p.parse_args()
    if not args.legacy_h5 and not args.factored_h5:
        print("Provide at least one of --legacy-h5 or --factored-h5", file=sys.stderr)
        return 2
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("Requires h5py", file=sys.stderr)
        return 1
    if args.legacy_h5:
        legacy_report(args.legacy_h5)
        print()
    if args.factored_h5:
        factored_report(args.factored_h5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
