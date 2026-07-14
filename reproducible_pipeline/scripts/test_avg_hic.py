#!/usr/bin/env python
"""Self-test for the ABC average-Hi-C loader + splitter (no external data needed).

Builds tiny synthetic single-file average-Hi-C tables (4-column intra and
8-column bedpe), runs scripts/split_avg_hic.py to fan them out per chromosome,
loads them via preprocessing.abc.contact.load_avg_hic, and checks that
get_hic_for_pred() returns the expected 5 kb bin-pair contacts and that it drops
into get_contacts_for_pairs() exactly like the .hic path.

Run:  python scripts/test_avg_hic.py     (prints PASS/FAIL, exits nonzero on failure)
"""
import gzip
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.abc.contact import load_avg_hic, get_contacts_for_pairs

RES = 5000
HERE = os.path.dirname(os.path.abspath(__file__))
SPLITTER = os.path.join(HERE, "split_avg_hic.py")

failures = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


def write_gz(path, lines):
    with gzip.open(path, "wt") as f:
        f.writelines(l if l.endswith("\n") else l + "\n" for l in lines)


def split(inp, out):
    subprocess.run([sys.executable, SPLITTER, "--in", inp, "--out", out],
                   check=True, capture_output=True, text=True)


def main():
    tmp = tempfile.mkdtemp(prefix="avghic_test_")

    # bin starts: bin i covers [i*RES, (i+1)*RES). Contacts: (chr, x1, x2) -> value
    # chr1: bins (0,0)=10.0, (0,2)=3.0, (1,3)=7.5 ; chr2: (5,5)=2.0
    intra_rows = [
        f"chr1\t{0*RES}\t{0*RES}\t10.0",
        f"chr1\t{0*RES}\t{2*RES}\t3.0",
        f"chr1\t{1*RES}\t{3*RES}\t7.5",
        f"chr2\t{5*RES}\t{5*RES}\t2.0",
    ]
    inp4 = os.path.join(tmp, "avg4.bed.gz"); write_gz(inp4, intra_rows)
    out4 = os.path.join(tmp, "by_chrom4"); split(inp4, out4)

    check("split produced chr1 + chr2 files",
          os.path.exists(os.path.join(out4, "chr1.tsv.gz")) and
          os.path.exists(os.path.join(out4, "chr2.tsv.gz")))

    hic = load_avg_hic(out4, resolution=RES)

    # pred: element_mid, tss -> expected contact
    # enhancer in bin2 (mid 2*RES+100), tss in bin0 (100) -> bin-pair (0,2)=3.0
    # enhancer in bin3, tss in bin1 -> (1,3)=7.5
    # enhancer in bin0, tss in bin0 -> (0,0)=10.0
    # enhancer in bin4, tss in bin0 -> absent -> NaN
    pred = pd.DataFrame({
        "element_mid": [2 * RES + 100, 3 * RES + 50, 100, 4 * RES + 10],
        "tss":         [100,           1 * RES + 20, 200, 100],
        "chrom":       ["chr1"] * 4,
    })
    got = hic.get_hic_for_pred(pred.copy(), "chr1")["hic_contact"].tolist()
    exp = [3.0, 7.5, 10.0, np.nan]
    ok = all((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9 for a, b in zip(got, exp))
    check(f"4-col lookup (got={got} exp={exp})", ok)

    # chr2 self-bin
    p2 = pd.DataFrame({"element_mid": [5 * RES + 300], "tss": [5 * RES + 10], "chrom": ["chr2"]})
    g2 = hic.get_hic_for_pred(p2, "chr2")["hic_contact"].tolist()
    check(f"chr2 self-bin (got={g2} exp=[2.0])", abs(g2[0] - 2.0) < 1e-9)

    # missing chromosome -> all NaN (power-law fallback downstream), no crash
    p3 = pd.DataFrame({"element_mid": [100], "tss": [200], "chrom": ["chr9"]})
    g3 = hic.get_hic_for_pred(p3, "chr9")["hic_contact"].tolist()
    check("missing chrom -> NaN", np.isnan(g3[0]))

    # 8-column bedpe input: chr1 x1 x2 chr2 y1 y2 name hic_contact ; one inter row dropped
    bedpe_rows = [
        f"chr1\t{0*RES}\t{1*RES}\t chr1\t{2*RES}\t{3*RES}\tp1\t4.0".replace(" ", ""),
        f"chr1\t{1*RES}\t{2*RES}\tchr2\t{0*RES}\t{1*RES}\tp2\t9.9",   # inter -> dropped
        f"chr1\t{6*RES}\t{7*RES}\tchr1\t{6*RES}\t{7*RES}\tp3\t5.0",
    ]
    inp8 = os.path.join(tmp, "avg8.bed.gz"); write_gz(inp8, bedpe_rows)
    out8 = os.path.join(tmp, "by_chrom8"); split(inp8, out8)
    # count lines in chr1 (should be 2 kept, inter dropped; no chr2 file)
    with gzip.open(os.path.join(out8, "chr1.tsv.gz"), "rt") as f:
        n1 = sum(1 for _ in f)
    check(f"8-col: intra kept=2, inter dropped (got {n1})", n1 == 2)
    check("8-col: no chr2 file (inter-only) ", not os.path.exists(os.path.join(out8, "chr2.tsv.gz")))
    hic8 = load_avg_hic(out8, resolution=RES)
    pb = pd.DataFrame({"element_mid": [2 * RES + 5], "tss": [0 * RES + 5], "chrom": ["chr1"]})
    gb = hic8.get_hic_for_pred(pb, "chr1")["hic_contact"].tolist()  # bins (0,2)=4.0
    check(f"8-col lookup (got={gb} exp=[4.0])", abs(gb[0] - 4.0) < 1e-9)

    # integration: get_contacts_for_pairs uses the avg map exactly like .hic
    pairs = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "element_mid": [2 * RES + 100, 4 * RES + 10],
        "tss": [100, 100],
        "distance": [2 * RES, 4 * RES],
    })
    out = get_contacts_for_pairs(pairs, hic_data=hic, resolution=RES,
                                 scale_hic_using_powerlaw=True)
    have_cols = all(c in out.columns for c in
                    ["hic_contact", "hic_contact_pl_scaled_adj", "powerlaw_contact"])
    finite = np.isfinite(out["hic_contact_pl_scaled_adj"]).all()
    check("get_contacts_for_pairs integrates avg map (cols + finite adj)", have_cols and finite)

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s) failed: {failures})")
        sys.exit(1)
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()
