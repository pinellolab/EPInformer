#!/usr/bin/env python3
"""Compare BSCC_GPU ABC `EnhancerPredictionsAllPutative*.txt` to this repo's `gene_enhancer_pair_all.csv`.

Reports per-gene link counts and nearest enhancer within --max-distance-bp for each TargetGene/ENSID.

Typical BSCC path (example):
  .../K562_*_ABC_nominated/Gene-enhancer links/EnhancerPredictionsAllPutative.avghic.txt
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    import numpy as np
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--old-abc",
        required=True,
        help="BSCC_GPU EnhancerPredictionsAllPutative*.txt (tab)",
    )
    p.add_argument(
        "--new-csv",
        default="batch_output/K562/encoding/gene_enhancer_pair_all.csv",
        help="Encoding-stage gene_enhancer_pair_all.csv",
    )
    p.add_argument(
        "--genes",
        nargs="+",
        required=True,
        metavar="ENSG",
        help="Target genes (ENSG IDs as in both tables)",
    )
    p.add_argument("--max-distance-bp", type=float, default=100_000.0)
    args = p.parse_args()
    max_bp = float(args.max_distance_bp)

    df_new = pd.read_csv(args.new_csv, low_memory=False)
    if "ENSID" not in df_new.columns or "distance_relative" not in df_new.columns:
        print("NEW csv missing ENSID / distance_relative", file=sys.stderr)
        return 1

    chunks = []
    want = set(args.genes)
    for ch in pd.read_csv(args.old_abc, sep="\t", chunksize=250_000, low_memory=False):
        m = ch["TargetGene"].isin(want)
        if m.any():
            chunks.append(ch.loc[m])
    old = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if old.empty:
        print("No rows in OLD file for requested genes.", file=sys.stderr)
        return 1

    print("OLD:", args.old_abc, "rows for genes:", len(old))
    print("NEW:", args.new_csv, "rows for genes:", int(df_new["ENSID"].isin(want).sum()))
    print(f"Window: |distance| <= {max_bp:.0f} bp\n")

    for ens in args.genes:
        sym = ""
        o = old[old["TargetGene"] == ens].copy()
        n = df_new[df_new["ENSID"] == ens].copy()
        print("=" * 72)
        print(ens)
        if o.empty:
            print("  OLD: (no rows)")
        else:
            o["abs_d"] = np.abs(o["distance"].astype(float))
            ow = o[o["abs_d"] <= max_bp]
            print(f"  OLD: links total={len(o)}  in_window={len(ow)}")
            if len(ow):
                r = ow.loc[ow["abs_d"].idxmin()]
                isp = r.get("isSelfPromoter", np.nan)
                print(
                    f"       nearest: |d|={r['abs_d']:.6g}  distance={r['distance']!r}  "
                    f"isSelfPromoter={isp}\n       name={r['name']}"
                )
        if n.empty:
            print("  NEW: (no rows)")
        else:
            n["abs_d"] = np.abs(n["distance_relative"].astype(float))
            nw = n[n["abs_d"] <= max_bp]
            print(f"  NEW: links total={len(n)}  in_window={len(nw)}")
            if len(nw):
                r = nw.loc[nw["abs_d"].idxmin()]
                print(
                    f"       nearest: |d|={r['abs_d']:.6g}  distance_relative={r['distance_relative']!r}\n"
                    f"       name={r['name']}"
                )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
