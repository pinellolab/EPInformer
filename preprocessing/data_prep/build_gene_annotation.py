#!/usr/bin/env python3
"""
Build a hg38 gene annotation BED from Roadmap's Ensembl v65 / Gencode v10
gene_info file by lifting over coordinates from hg19 → hg38.

This ensures the gene BED and expression CSV are derived from the same
annotation source (Roadmap Epigenomics).

Requires:
  - liftOver binary (UCSC) on PATH or at --liftover-bin
  - hg19ToHg38.over.chain(.gz) — auto-downloaded if not found

Usage::

    python preprocessing/data_prep/build_gene_annotation.py \\
        --gene-set pc \\
        --output-dir data/reference/hg38

    python preprocessing/data_prep/build_gene_annotation.py \\
        --gene-set pc_linc \\
        --output-dir data/reference/hg38
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
_ROADMAP_EXPR = "https://egg2.wustl.edu/roadmap/data/byDataType/rna/expression"
_GENE_INFO_URL = f"{_ROADMAP_EXPR}/Ensembl_v65.Gencode_v10.ENSG.gene_info"
_CHAIN_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"

# Gene types to include for each --gene-set option
_GENE_SETS = {
    "pc": {"protein_coding"},
    "pc_linc": {"protein_coding", "lincRNA"},
}


def _download(url: str, dest: str) -> str:
    """Download a file if not already cached."""
    if os.path.isfile(dest):
        print(f"  Cached: {dest}")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved: {dest}")
    return dest


def _find_liftover(user_bin: str | None) -> str:
    """Locate the liftOver binary."""
    if user_bin and os.path.isfile(user_bin):
        return user_bin
    found = shutil.which("liftOver")
    if found:
        return found
    print("ERROR: liftOver binary not found. Install from "
          "https://hgdownload.soe.ucsc.edu/admin/exe/", file=sys.stderr)
    sys.exit(1)


def _load_gene_info(path: str) -> list[dict]:
    """Parse the Ensembl v65 gene_info file into a list of dicts."""
    genes = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            chrom = parts[1]
            # Skip non-standard chromosomes (patches, haplotypes)
            if "_" in chrom or chrom.startswith("GL") or chrom.startswith("KI"):
                continue
            strand_num = int(parts[4])
            genes.append({
                "ensid": parts[0],
                "chrom": chrom if chrom.startswith("chr") else f"chr{chrom}",
                "start": int(parts[2]),
                "end": int(parts[3]),
                "strand": "+" if strand_num >= 0 else "-",
                "biotype": parts[5],
                "gene_name": parts[6] if parts[6] else parts[0],
            })
    return genes


def _run_liftover(
    genes: list[dict],
    liftover_bin: str,
    chain_file: str,
) -> list[dict]:
    """Liftover gene coordinates from hg19 → hg38 using UCSC liftOver."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f_in:
        in_bed = f_in.name
        for i, g in enumerate(genes):
            # BED format: chr start end name score strand
            f_in.write(f"{g['chrom']}\t{g['start']}\t{g['end']}\t{i}\t0\t{g['strand']}\n")

    out_bed = in_bed + ".lifted"
    unmapped = in_bed + ".unmapped"

    try:
        cmd = [liftover_bin, in_bed, chain_file, out_bed, unmapped]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"liftOver stderr: {result.stderr}", file=sys.stderr)

        # Parse lifted coordinates
        lifted = {}
        with open(out_bed) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                idx = int(parts[3])
                lifted[idx] = {
                    "chrom": parts[0],
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "strand": parts[5],
                }

        # Count unmapped
        n_unmapped = 0
        with open(unmapped) as f:
            for line in f:
                if not line.startswith("#"):
                    n_unmapped += 1

        print(f"  Lifted: {len(lifted)}/{len(genes)} genes "
              f"({n_unmapped} unmapped)")

        # Merge lifted coords back into gene records
        result_genes = []
        for i, g in enumerate(genes):
            if i in lifted:
                g_out = g.copy()
                g_out["chrom"] = lifted[i]["chrom"]
                g_out["start"] = lifted[i]["start"]
                g_out["end"] = lifted[i]["end"]
                g_out["strand"] = lifted[i]["strand"]
                result_genes.append(g_out)

        return result_genes

    finally:
        for f in [in_bed, out_bed, unmapped]:
            if os.path.isfile(f):
                os.unlink(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hg38 gene annotation BED from Roadmap Ensembl v65 gene_info"
    )
    parser.add_argument(
        "--gene-set",
        choices=list(_GENE_SETS.keys()),
        default="pc",
        help="Gene set to include: 'pc' (protein_coding only, ~20K) or "
             "'pc_linc' (protein_coding + lincRNA, ~25K). Default: pc.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/reference/hg38",
        help="Output directory for the BED file. Default: data/reference/hg38",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache downloaded files. Default: {output-dir}/.liftover_cache",
    )
    parser.add_argument(
        "--liftover-bin",
        default=None,
        help="Path to liftOver binary. Default: auto-detect from PATH.",
    )
    parser.add_argument(
        "--chain-file",
        default=None,
        help="Path to hg19ToHg38.over.chain(.gz). Default: auto-download.",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    cache_dir = args.cache_dir or os.path.join(out_dir, ".liftover_cache")
    os.makedirs(out_dir, exist_ok=True)

    # --- Locate tools ---
    liftover_bin = _find_liftover(args.liftover_bin)
    print(f"Using liftOver: {liftover_bin}")

    # --- Download files ---
    print("Downloading reference files ...")
    gene_info_path = _download(
        _GENE_INFO_URL,
        os.path.join(cache_dir, "Ensembl_v65.Gencode_v10.ENSG.gene_info"),
    )

    if args.chain_file and os.path.isfile(args.chain_file):
        chain_file = args.chain_file
    else:
        chain_file = os.path.join(cache_dir, "hg19ToHg38.over.chain.gz")
        _download(_CHAIN_URL, chain_file)

    # --- Load and filter genes ---
    print("Loading gene annotations ...")
    all_genes = _load_gene_info(gene_info_path)
    print(f"  Total genes parsed: {len(all_genes)}")

    allowed_types = _GENE_SETS[args.gene_set]
    filtered = [g for g in all_genes if g["biotype"] in allowed_types]
    print(f"  After filtering to {allowed_types}: {len(filtered)} genes")

    # --- Liftover hg19 → hg38 ---
    print("Running liftOver hg19 → hg38 ...")
    lifted = _run_liftover(filtered, liftover_bin, chain_file)

    # --- Deduplicate by ENSID (keep first occurrence) ---
    seen = set()
    deduped = []
    for g in lifted:
        if g["ensid"] not in seen:
            seen.add(g["ensid"])
            deduped.append(g)
    print(f"  After dedup: {len(deduped)} unique genes")

    # --- Sort by chromosome and position ---
    chrom_order = {f"chr{c}": i for i, c in enumerate(
        list(range(1, 23)) + ["X", "Y", "M"]
    )}
    deduped.sort(key=lambda g: (chrom_order.get(g["chrom"], 99), g["start"]))

    # --- Write BED ---
    out_name = f"CollapsedGeneBounds.Ensembl_v65.hg38.{args.gene_set}.bed"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w") as f:
        f.write("#chr\tstart\tend\tname\tscore\tstrand\tEnsembl_ID\tgene_type\n")
        for g in deduped:
            f.write(f"{g['chrom']}\t{g['start']}\t{g['end']}\t"
                    f"{g['gene_name']}\t0\t{g['strand']}\t"
                    f"{g['ensid']}\t{g['biotype']}\n")

    print(f"\nDone! {len(deduped)} genes written to {out_path}")

    # --- Summary by type ---
    from collections import Counter
    type_counts = Counter(g["biotype"] for g in deduped)
    for bt, cnt in type_counts.most_common():
        print(f"  {bt}: {cnt}")


if __name__ == "__main__":
    main()
