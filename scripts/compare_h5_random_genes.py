#!/usr/bin/env python3
"""Sample random genes present in both legacy and factored HDF5; compare per-gene enhancer features.

Legacy: enhancers_feat columns 0,3,4 match the 3-channel training path (abs dist, activity_base_no_qnorm, hic_contact).
Factored: distance, activity, contact for slots with gene_enh_idx >= 0.

Use --max-distance-bp (default 100000) to restrict to the same TSS-centered window as factored encoding (config max_distance).

With --global-only, print only dataset-level in-window counts and factored contact NaN rate.

With --include-global-stats, print that dataset block before the random-gene section (default: random genes only).
"""

from __future__ import annotations

import argparse
import sys


def _decode_ensid(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _legacy_used_mask(ef_row: "np.ndarray", dist_eps: float = 1e-6) -> "np.ndarray":
    import numpy as np

    d = np.abs(ef_row[:, 0])
    return (d > dist_eps) | (np.abs(ef_row[:, 1:]).max(axis=1) > dist_eps)


def _print_global_stats(
    enh_feat: "np.ndarray",
    gene_enh_idx: "np.ndarray",
    dist: "np.ndarray",
    contact_raw: "np.ndarray",
    max_bp: float,
) -> None:
    """Vectorized dataset-level stats for |distance| <= max_bp."""
    import numpy as np

    d_leg = np.abs(enh_feat[..., 0])
    used = (d_leg > 1e-6) | (np.abs(enh_feat[..., 1:]).max(axis=-1) > 1e-6)
    leg_win = used & (d_leg <= max_bp)
    n_leg_per_gene = leg_win.sum(axis=1)
    total_leg = int(leg_win.sum())

    valid = gene_enh_idx >= 0
    d_fac = np.abs(dist)
    fac_win = valid & (d_fac <= max_bp)
    n_fac_per_gene = fac_win.sum(axis=1)
    total_fac = int(fac_win.sum())

    nan_contact = np.isnan(contact_raw) & fac_win
    n_nan = int(nan_contact.sum())
    denom = int(fac_win.sum())
    pct = 100.0 * n_nan / max(denom, 1)

    print("=== Dataset-level (|distance| <= {:.0f} bp) ===".format(max_bp))
    print(f"Legacy:   total in-window enhancer rows (sum over genes): {total_leg}")
    print(f"          per-gene count: min={n_leg_per_gene.min()} median={float(np.median(n_leg_per_gene)):.1f} "
          f"mean={n_leg_per_gene.mean():.2f} max={n_leg_per_gene.max()}")
    print(f"Factored: total in-window valid slots (sum over genes): {total_fac}")
    print(f"          per-gene count: min={n_fac_per_gene.min()} median={float(np.median(n_fac_per_gene)):.1f} "
          f"mean={n_fac_per_gene.mean():.2f} max={n_fac_per_gene.max()}")
    print(f"Factored: NaN in contact (restricted to in-window valid slots): {n_nan} / {denom} ({pct:.4f}%)")


def main() -> int:
    import numpy as np
    import h5py

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-h5", required=True)
    p.add_argument("--factored-h5", required=True)
    p.add_argument("--n-genes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--genes",
        nargs="+",
        default=None,
        metavar="ENSG",
        help="Explicit gene_id list (overrides random --n-genes / --seed).",
    )
    p.add_argument(
        "--max-distance-bp",
        type=float,
        default=100_000.0,
        help="Only enhancers with abs(distance) <= this value (bp). Default 100000 matches factored max_distance.",
    )
    p.add_argument(
        "--global-only",
        action="store_true",
        help="Print only dataset-level in-window stats; skip random gene sampling.",
    )
    p.add_argument(
        "--include-global-stats",
        action="store_true",
        help="Also print dataset-level block before random genes (not implied by default).",
    )
    args = p.parse_args()
    max_bp = float(args.max_distance_bp)

    with h5py.File(args.legacy_h5, "r") as f:
        leg_ens = [_decode_ensid(x) for x in f["ensid"][:]]
        enh_feat = np.asarray(f["enhancers_feat"][:], dtype=np.float64)

    with h5py.File(args.factored_h5, "r") as f:
        fac_ens = [_decode_ensid(x) for x in f["ensid"][:]]
        gene_enh_idx = np.asarray(f["gene_enh_idx"][:], dtype=np.int64)
        dist = np.asarray(f["distance"][:], dtype=np.float64)
        contact_raw = np.asarray(f["contact"][:], dtype=np.float64)
        activity = np.nan_to_num(np.asarray(f["activity"][:], dtype=np.float64), nan=0.0)
        contact = np.nan_to_num(contact_raw.copy(), nan=0.0)

    print(f"Legacy:   {args.legacy_h5}")
    print(f"Factored: {args.factored_h5}")
    print(f"Window:   abs(distance) <= {max_bp:.0f} bp\n")

    if args.global_only:
        _print_global_stats(enh_feat, gene_enh_idx, dist, contact_raw, max_bp)
        return 0

    if args.include_global_stats:
        _print_global_stats(enh_feat, gene_enh_idx, dist, contact_raw, max_bp)
        print()

    leg_set = set(leg_ens)
    fac_set = set(fac_ens)
    common = sorted(leg_set & fac_set)
    if not common:
        print("No overlapping ensid between files.", file=sys.stderr)
        return 1

    leg_idx = {e: i for i, e in enumerate(leg_ens)}
    fac_idx = {e: i for i, e in enumerate(fac_ens)}

    if args.genes:
        picked = [g.strip() for g in args.genes]
        missing = [g for g in picked if g not in leg_set or g not in fac_set]
        if missing:
            print(f"Unknown or not in both HDF5: {missing}", file=sys.stderr)
            return 2
    else:
        rng = np.random.default_rng(args.seed)
        k = min(args.n_genes, len(common))
        picked = [str(x) for x in rng.choice(common, size=k, replace=False)]

    print(f"Common genes: {len(common)} / legacy {len(leg_ens)} / factored {len(fac_ens)}")
    if args.genes:
        print(f"Requested genes: {picked}\n")
    else:
        print(f"Random sample (seed={args.seed}, n={len(picked)}): {picked}\n")

    for ensid in picked:
        li = leg_idx[ensid]
        fi = fac_idx[ensid]
        ef = enh_feat[li]
        used = _legacy_used_mask(ef)
        in_win_leg = used & (np.abs(ef[:, 0]) <= max_bp)
        n_leg = int(in_win_leg.sum())

        idx = gene_enh_idx[fi]
        valid = idx >= 0
        dist_f = dist[fi]
        act_f = activity[fi]
        con_f = contact[fi]
        in_win_fac = valid & (np.abs(dist_f) <= max_bp)
        n_fac = int(in_win_fac.sum())

        leg_m = np.column_stack([np.abs(ef[:, 0]), ef[:, 3], ef[:, 4]])
        leg_m = leg_m[in_win_leg]
        order = np.argsort(leg_m[:, 0]) if leg_m.shape[0] else np.array([], dtype=np.int64)
        leg_sorted = leg_m[order] if leg_m.shape[0] else leg_m

        fac_m = np.column_stack([np.abs(dist_f), act_f, con_f])
        fac_m = fac_m[in_win_fac]
        order_f = np.argsort(fac_m[:, 0]) if fac_m.shape[0] else np.array([], dtype=np.int64)
        fac_sorted = fac_m[order_f] if fac_m.shape[0] else fac_m

        print(f"=== {ensid} ===")
        print(f"  legacy in-window rows: {n_leg} / 200 (non-pad and |d|<={max_bp:.0f})")
        print(f"  factored in-window slots: {n_fac} / {gene_enh_idx.shape[1]}")

        tk = args.top_k
        print(f"  top {tk} by abs(distance) — legacy [abs_d, col3, col4]:")
        for j in range(min(tk, leg_sorted.shape[0])):
            print(f"    {j}:  {leg_sorted[j, 0]:.4g}  {leg_sorted[j, 1]:.6g}  {leg_sorted[j, 2]:.6g}")
        print(f"  top {tk} by abs(distance) — factored [abs_d, activity, contact]:")
        for j in range(min(tk, fac_sorted.shape[0])):
            print(f"    {j}:  {fac_sorted[j, 0]:.4g}  {fac_sorted[j, 1]:.6g}  {fac_sorted[j, 2]:.6g}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
