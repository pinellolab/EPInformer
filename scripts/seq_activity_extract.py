#!/usr/bin/env python
"""Encoder sequence+activity extraction.

Replicates process_EPInformer_data_v2.ipynb cells 35 (5-bin construction) and
38 (sequence + activity extraction) exactly:

  * summit  = narrowPeak.start + col10 (peak offset)
  * 5 bins (256 bp) at summit + 192*{-2,-1,0,1,2}, built with the exact
    slop arithmetic (center summit+-128; right1 [summit+64, summit+320]; ...)
  * per-rep RPM = 1e6 * pysam.count(chr,start,end) / mapped_reads (raw counts, NO read extension)
  * DHS.RPM / H3K27ac.RPM pooled over >=1 replicate BAM per assay via --pool-method:
      mean = 1/N * Sum(per-rep RPM)   <- the exact recipe (2 reps/assay); DEFAULT
      sum  = 1e6 * Sum(count)/Sum(mapped)  (depth-weighted; == mean for a single rep)
  * Activity = sqrt(DHS.RPM * H3K27ac.RPM)  (linear; log2 applied at train time)
  * sequence = fasta[chr, start:end] upper, N-padded at chrom edges
  * NO negative samples (pure peak bins -> Npeaks*5 rows)

Output columns match our data/enhancer_sequences schema so train_seqEncoder.py
can consume it via --data-csv.

Usage (GM12878 — reaches 0.617: 2 filtered reps/assay, mean-pool; Activity byte-identical):
  python scripts/seq_activity_extract.py \
      --narrowpeak reference/GM12878_H3K27ac.ENCFF023LTU.narrowPeak \
      --dnase-bam data/GM12878/DNase/ENCFF729UYK.bam data/GM12878/DNase/ENCFF020WZB.bam \
      --h3k27ac-bam data/GM12878/H3K27ac/ENCFF269GKF.bam data/GM12878/H3K27ac/ENCFF201OHW.bam \
      --pool-method mean --fasta data/reference/hg38/hg38.fa \
      --cell GM12878 --out batch_output/GM12878/links/GM12878_peak_5bins_around_summit_activity_sequence.csv
  # K562 needs only single reps (mean == sum for one BAM): --dnase-bam ...ENCFF257HEE.bam --h3k27ac-bam ...ENCFF232RQF.bam
"""
import argparse
import numpy as np
import pandas as pd
import pysam

_NP_COLS = ["chr", "start", "end", "name", "score", "strand",
            "signalValue", "pValue", "qValue", "peak"]
# the reference cell-35 bins: (Pos, Start_offset_from_summit, End_offset_from_summit)
_BINS = [(0, -128, 128), (1, 64, 320), (2, 256, 512), (-1, -320, -64), (-2, -512, -256)]


def _seq(fa, chrom, start, end, clen):
    # the reference FastaStringExtractor: truncate to [0,clen], pad with N (kipoiseq Interval)
    s, e = max(start, 0), min(end, clen)
    seq = fa.fetch(chrom, s, e).upper() if e > s else ""
    return "N" * max(-start, 0) + seq + "N" * max(end - clen, 0)


def _count(b, chrom, start, end, ext, clen):
    """Count reads overlapping [start, end). If ext>0, first extend each read to
    ext bp from its 5' end (ABC/MACS single-end extension) — +strand read ->
    [5'start, 5'start+ext], -strand read -> [3'end-ext, 3'end] — then test overlap.
    ext<=0 = raw pysam.count overlap (the shipped compute_activity default)."""
    if ext <= 0:
        return b.count(chrom, start, end)
    lo, hi = max(0, start - ext), min(clen, end + ext)
    n = 0
    for r in b.fetch(chrom, lo, hi):
        if r.is_unmapped:
            continue
        if r.is_reverse:
            fs, fe = r.reference_end - ext, r.reference_end
        else:
            fs, fe = r.reference_start, r.reference_start + ext
        if fs < end and fe > start:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--narrowpeak", required=True, help="H3K27ac narrowPeak (summit source)")
    ap.add_argument("--dnase-bam", nargs="+", required=True,
                    help="one or more DNase BAMs; multiple are pooled (counts+totals summed)")
    ap.add_argument("--h3k27ac-bam", nargs="+", required=True,
                    help="one or more H3K27ac BAMs; multiple are pooled (counts+totals summed)")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--cell", default="GM12878")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extend", type=int, default=0,
                    help="ABC/MACS-style single-end read extension to N bp before counting "
                         "(0 = raw pysam.count overlap; try 200 to fill sparse flank bins)")
    ap.add_argument("--pool-method", choices=["sum", "mean"], default="mean",
                    help="how to pool multiple replicate BAMs (default: mean = the exact "
                         "recipe). 'mean' = average of per-rep RPMs 1/N·Σ(1e6*count_i/mapped_i) "
                         "(H3K27ac_RPM = mean(H3K27ac_0_RPM, H3K27ac_1_RPM); reproduces GM12878 "
                         "0.617). 'sum' = 1e6*Σcount/Σmapped (depth-weighted). Identical for a "
                         "single rep (e.g. K562).")
    args = ap.parse_args()

    peaks = pd.read_csv(args.narrowpeak, sep="\t", header=None, names=_NP_COLS, comment="#")
    peaks["summit"] = peaks["start"].astype(int) + peaks["peak"].astype(int)
    print(f"[{args.cell}] {len(peaks)} peaks from {args.narrowpeak}")

    fa = pysam.FastaFile(args.fasta)
    clens = dict(zip(fa.references, fa.lengths))
    bds = [pysam.AlignmentFile(p) for p in args.dnase_bam]
    bhs = [pysam.AlignmentFile(p) for p in args.h3k27ac_bam]
    dn_chrs = [set(b.references) for b in bds]
    h3_chrs = [set(b.references) for b in bhs]
    dn_maps = [b.mapped for b in bds]
    h3_maps = [b.mapped for b in bhs]
    print(f"mapped reads per rep: DNase={dn_maps} H3K27ac={h3_maps}  pool={args.pool_method}")

    def _rpm(bams, maps, chrs, chrom, start, end):
        """Per-bin RPM pooled across replicate BAMs (skipping bams lacking the chrom)."""
        cm = [(_count(b, chrom, start, end, args.extend, clen), m)
              for b, m, cs in zip(bams, maps, chrs) if chrom in cs]
        if not cm:
            return 0.0
        if args.pool_method == "sum":          # depth-weighted: 1e6*Σcount/Σmapped
            tc, tm = sum(c for c, _ in cm), sum(m for _, m in cm)
            return 1e6 * tc / tm
        return sum(1e6 * c / m for c, m in cm) / len(cm)   # mean of per-rep RPMs

    rows = []
    for pi, r in enumerate(peaks.itertuples(index=False)):
        chrom, summit, name = r.chr, int(r.summit), r.name
        clen = clens.get(chrom)
        if clen is None:
            continue
        for pos, so, eo in _BINS:
            start, end = summit + so, summit + eo
            if start < 0 or end > clen:
                continue
            seq = _seq(fa, chrom, start, end, clen)
            dn_rpm = _rpm(bds, dn_maps, dn_chrs, chrom, start, end)
            h3_rpm = _rpm(bhs, h3_maps, h3_chrs, chrom, start, end)
            rows.append((name, chrom, start, end, summit, pos,
                         dn_rpm, h3_rpm, seq, float(np.sqrt(dn_rpm * h3_rpm))))
        if pi % 10000 == 0:
            print(f"  {pi}/{len(peaks)} peaks", flush=True)

    out = pd.DataFrame(rows, columns=[
        "Name", "Chromosome", "Start", "End", "Summit", "Offset_to_summit",
        "DNase_RPM", "H3K27ac_RPM", "Sequence", "Activity"])
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    z = 100 * (out["Activity"] == 0).mean()
    print(f"wrote {len(out)} rows ({out['Name'].nunique()} peaks) zeroActivity={z:.2f}%  -> {args.out}")


if __name__ == "__main__":
    main()
