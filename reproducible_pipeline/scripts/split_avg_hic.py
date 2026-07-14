#!/usr/bin/env python
"""Split the single ABC / ENCODE-rE2G *average* Hi-C file into a per-chromosome
directory consumable by ``preprocessing.abc.contact.AverageHiCContactMap``.

The ENCODE annotation ENCSR382HAW ships the ABC average Hi-C (5 kb, GRCh38,
averaged over 34 Hi-C datasets) as ONE large file: ``ENCFF134PUN.bed.gz`` (~58 GB).
That is too big to hold in memory, so we stream it once and fan out one gzip
writer per chromosome, producing the per-chromosome layout the Broad ABC pipeline
uses for ``--hic_type avg``.

Input (headerless, TAB-separated) — auto-detected by column count:
  * 4 columns:  ``chr  x1  x2  hic_contact``                    (intra-chromosomal)
  * 8 columns:  ``chr1 x1 x2 chr2 y1 y2 name hic_contact``      (bedpe; intra rows kept)
  * N>=4 cols:  assume ``chr, x1, x2, ..., <last>=hic_contact`` (best-effort)

``x1``/``x2`` (or ``x1``/``y1`` for bedpe) are the *start* positions (bp) of the two
5 kb bins; the loader re-bins them by resolution.

Output:  ``{out_dir}/{chrom}.tsv.gz``  with columns  ``x1 <TAB> x2 <TAB> hic_contact``.

Only intra-chromosomal contacts are written (ABC scores intra-chromosomal
enhancer–TSS pairs); pass ``--keep-inter`` to retain inter-chromosomal rows too
(they are ignored by the loader regardless).

Usage:
  python scripts/split_avg_hic.py --in data/reference/abc_avg_hic/ENCFF134PUN.bed.gz \
                                  --out data/reference/abc_avg_hic/by_chrom
  # then point config hic_file (or --hic) at  data/reference/abc_avg_hic/by_chrom
"""
import argparse
import gzip
import os
import sys


def _open_text(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="single average-Hi-C file (.gz ok)")
    ap.add_argument("--out", dest="out", required=True, help="output directory for per-chromosome files")
    ap.add_argument("--keep-inter", action="store_true",
                    help="also write inter-chromosomal rows (default: drop them)")
    ap.add_argument("--progress", type=int, default=50_000_000,
                    help="print a progress line every N input rows (default 50M)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    writers = {}
    ncol = None
    total = kept = skipped_inter = 0

    with _open_text(args.inp, "rt") as f:
        for line in f:
            if not line or line[0] == "#" or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if ncol is None:
                ncol = len(parts)
                if ncol == 8:
                    fmt = "bedpe (chr1 x1 x2 chr2 y1 y2 name hic_contact)"
                elif ncol == 4:
                    fmt = "intra (chr x1 x2 hic_contact)"
                elif ncol >= 4:
                    fmt = f"best-effort {ncol}-col (chr x1 x2 ... last=hic_contact)"
                else:
                    sys.exit(f"ERROR: need >=4 columns, first data row has {ncol}: {parts}")
                print(f"[split_avg_hic] detected format: {fmt}", file=sys.stderr)

            total += 1
            if ncol == 8:
                c1, x1, x2, c2, y1, y2 = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                contact = parts[7]
                if c1 != c2:
                    if not args.keep_inter:
                        skipped_inter += 1
                        continue
                chrom, ox1, ox2 = c1, x1, y1          # two bin start positions
            else:
                chrom, ox1, ox2, contact = parts[0], parts[1], parts[2], parts[-1]

            w = writers.get(chrom)
            if w is None:
                w = gzip.open(os.path.join(args.out, f"{chrom}.tsv.gz"), "wt")
                writers[chrom] = w
            w.write(f"{ox1}\t{ox2}\t{contact}\n")
            kept += 1

            if args.progress and total % args.progress == 0:
                print(f"[split_avg_hic] {total:,} rows read, {kept:,} written, "
                      f"{len(writers)} chroms ...", file=sys.stderr)

    for w in writers.values():
        w.close()
    print(f"[split_avg_hic] DONE: total={total:,} kept={kept:,} "
          f"skipped_inter={skipped_inter:,} chroms={len(writers)} -> {args.out}")
    if writers:
        print("  chroms:", ", ".join(sorted(writers)))


if __name__ == "__main__":
    main()
