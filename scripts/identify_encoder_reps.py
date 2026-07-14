#!/usr/bin/env python
"""Identify what the reference H3K27ac_0/_1 and DNase_0/_1 columns actually are, by matching
their per-bin RPM values against RPMs recomputed from candidate ENCODE BAMs.

For a stride-sample of reference-CSV bins, compute RPM = 1e6*count/mapped from each candidate
BAM and correlate (+ median ratio) against each reference column. A near-1.0 corr AND ~1.0 ratio
pins the column to that exact BAM -> tells us bio-rep vs tech-rep and filtered vs unfiltered.
"""
import numpy as np, pandas as pd, pysam, glob, os

REF_CSV = "/path/to/enhancer_sequences/GM12878_peak_5bins_around_summit_activity_sequence.csv"
DN = "data/GM12878/DNase"; H3 = "data/GM12878/H3K27ac"
# candidate BAMs present on disk (filtered + any unfiltered already downloaded)
CANDS = {
  "H3K27ac_ENCFF269GKF_filt_rep1": f"{H3}/ENCFF269GKF.bam",
  "H3K27ac_ENCFF201OHW_filt_rep2": f"{H3}/ENCFF201OHW.bam",
  "H3K27ac_ENCFF620IUQ_unfilt_rep1": f"{H3}/ENCFF620IUQ.bam",
  "H3K27ac_ENCFF067FBS_unfilt_rep2": f"{H3}/ENCFF067FBS.bam",
  "DNase_ENCFF729UYK_filt_rep2": f"{DN}/ENCFF729UYK.bam",
  "DNase_ENCFF020WZB_filt_rep1": f"{DN}/ENCFF020WZB.bam",
  "DNase_ENCFF467CXY_unfilt_rep1": f"{DN}/ENCFF467CXY.bam",
  "DNase_ENCFF940NSD_unfilt_rep2": f"{DN}/ENCFF940NSD.bam",
}

df = pd.read_csv(REF_CSV, usecols=["Chromosome","Start","End","H3K27ac_0_RPM","H3K27ac_1_RPM","DNase_0_RPM","DNase_1_RPM"])
samp = df.iloc[::100].reset_index(drop=True)      # ~2684 bins spread across the file
print(f"sampled {len(samp)} bins from {len(df)}")

def rpm_for(bam):
    if not os.path.exists(bam):
        return None
    b = pysam.AlignmentFile(bam)
    if b.check_index() is False:
        return None
    chrs = set(b.references)
    mapped = sum(int(x.split('\t')[2]) for x in pysam.idxstats(bam).strip().split('\n'))
    out = np.zeros(len(samp))
    for i, r in enumerate(samp.itertuples(index=False)):
        c = r.Chromosome
        out[i] = 1e6 * b.count(c, int(r.Start), int(r.End)) / mapped if c in chrs else 0.0
    return out, mapped

cols = {k: samp[k].values for k in ["H3K27ac_0_RPM","H3K27ac_1_RPM","DNase_0_RPM","DNase_1_RPM"]}
for name, bam in CANDS.items():
    res = rpm_for(bam)
    if res is None:
        print(f"{name:38s} -- BAM missing/unindexed (skip)")
        continue
    vals, mapped = res
    print(f"\n{name}  (mapped={mapped:,})")
    for cname, cvals in cols.items():
        if not cname.startswith(name.split('_')[0]):
            continue  # only compare H3K27ac candidate vs H3K27ac cols, etc.
        m = (cvals > 0) | (vals > 0)
        r = np.corrcoef(vals[m], cvals[m])[0, 1] if m.sum() > 2 else float('nan')
        nz = vals[cvals > 0]
        ratio = np.median(nz[nz > 0] / cvals[cvals > 0][nz > 0]) if (cvals > 0).any() else float('nan')
        print(f"    vs {cname:16s} corr={r:.4f}  median(bam/ref)={ratio:.3f}")
