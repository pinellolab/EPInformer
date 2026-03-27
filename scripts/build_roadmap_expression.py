#!/usr/bin/env python3
"""
One-time script: download Roadmap Epigenomics RNA-seq expression data
for all 57 epigenomes and build expression CSVs compatible with EPInformer.

Downloads from:
  https://egg2.wustl.edu/roadmap/data/byDataType/rna/expression/

Outputs:
  {output_dir}/roadmap_57epigenomes_rpkm.csv          — raw RPKM matrix (genes × epigenomes)
  {output_dir}/roadmap_expression_all.csv              — merged with gene features (for EPInformer)
  {output_dir}/roadmap_epigenome_metadata.csv          — epigenome ID → name mapping

Usage::

    python scripts/build_roadmap_expression.py \
        --xpresso-csv data/GM12878_K562_18377_gene_expr_fromXpresso.csv \
        --output-dir data/roadmap_expression

    # Use custom cache directory for downloaded files
    python scripts/build_roadmap_expression.py \
        --xpresso-csv data/GM12878_K562_18377_gene_expr_fromXpresso.csv \
        --output-dir data/roadmap_expression \
        --cache-dir /tmp/roadmap_cache
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Roadmap URLs
# ---------------------------------------------------------------------------
_BASE = "https://egg2.wustl.edu/roadmap/data/byDataType/rna/expression"
_RPKM_PC_URL = f"{_BASE}/57epigenomes.RPKM.pc.gz"
_GENE_INFO_URL = f"{_BASE}/Ensembl_v65.Gencode_v10.ENSG.gene_info"
_EG_NAME_URL = f"{_BASE}/EG.name.txt"

# Known Roadmap ID → common cell-type name mappings
_ROADMAP_CELL_NAMES = {
    "E003": "H1",
    "E114": "A549",
    "E116": "GM12878",
    "E117": "HeLa",
    "E118": "HepG2",
    "E122": "HUVEC",
    "E123": "K562",
    "E127": "NHEK",
}


def _download(url: str, cache_dir: str) -> str:
    """Download a file to cache_dir, return local path. Skip if exists."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = url.rsplit("/", 1)[-1]
    local = os.path.join(cache_dir, fname)
    if os.path.isfile(local):
        print(f"  Cached: {local}")
        return local
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, local)
    print(f"  Saved: {local}")
    return local


def _load_rpkm(path: str) -> pd.DataFrame:
    """Load the gzipped RPKM matrix (genes × epigenomes)."""
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    # Index is ENSID (e.g. ENSG00000000003), columns are epigenome IDs (e.g. E003)
    df.index.name = "ENSID"
    return df


def _load_gene_info(path: str) -> pd.DataFrame:
    """Load the Ensembl v65 / Gencode v10 gene annotation."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["ENSID", "chrom", "start", "end", "strand_num", "biotype",
               "gene_name", "description"],
    )
    # Convert strand: 1 → "+", -1 → "-"
    df["strand"] = df["strand_num"].map({1: "+", -1: "-"})
    return df


def _load_eg_names(path: str) -> pd.DataFrame:
    """Load the epigenome ID → name mapping."""
    df = pd.read_csv(path, sep="\t", header=None, names=["epigenome_id", "name"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Roadmap expression data and build EPInformer-compatible CSVs"
    )
    parser.add_argument(
        "--xpresso-csv",
        default=None,
        help="Path to existing Xpresso gene features CSV "
             "(e.g. data/GM12878_K562_18377_gene_expr_fromXpresso.csv). "
             "If provided, gene features (UTR, CDS, coordinates) are merged in.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/roadmap_expression",
        help="Output directory for generated CSVs.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache downloaded files (default: {output-dir}/.cache).",
    )
    parser.add_argument(
        "--log-transform",
        choices=["log10_xpresso", "log2", "none"],
        default="log10_xpresso",
        help="Transformation for RPKM → expression target. "
             "Default: log10(RPKM + 0.1) matching Xpresso convention.",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    cache_dir = args.cache_dir or os.path.join(out_dir, ".cache")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download Roadmap files
    # ------------------------------------------------------------------
    print("Downloading Roadmap expression data ...")
    rpkm_path = _download(_RPKM_PC_URL, cache_dir)
    gene_info_path = _download(_GENE_INFO_URL, cache_dir)
    eg_name_path = _download(_EG_NAME_URL, cache_dir)

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print("Loading RPKM matrix ...")
    rpkm = _load_rpkm(rpkm_path)
    print(f"  {rpkm.shape[0]} genes × {rpkm.shape[1]} epigenomes")

    gene_info = _load_gene_info(gene_info_path)
    eg_names = _load_eg_names(eg_name_path)

    # Save metadata
    meta_path = os.path.join(out_dir, "roadmap_epigenome_metadata.csv")
    eg_names.to_csv(meta_path, index=False)
    print(f"  Metadata → {meta_path}")

    # ------------------------------------------------------------------
    # 3. Save raw RPKM matrix
    # ------------------------------------------------------------------
    rpkm_out = os.path.join(out_dir, "roadmap_57epigenomes_rpkm.csv")
    rpkm.to_csv(rpkm_out)
    print(f"  Raw RPKM → {rpkm_out}")

    # ------------------------------------------------------------------
    # 4. Apply log transformation → Actual_{cell_type} columns
    # ------------------------------------------------------------------
    # Filter to Xpresso gene list if provided (so z-score stats match)
    if args.xpresso_csv:
        xpresso_path = os.path.abspath(args.xpresso_csv)
        if os.path.isfile(xpresso_path):
            xp_genes = set(pd.read_csv(xpresso_path, usecols=["ENSID"])["ENSID"])
            rpkm_subset = rpkm.loc[rpkm.index.isin(xp_genes)]
            print(f"  Filtered to {len(rpkm_subset)}/{len(rpkm)} Xpresso genes for z-score")
        else:
            rpkm_subset = rpkm
    else:
        rpkm_subset = rpkm

    if args.log_transform == "log10_xpresso":
        log_all = np.log10(rpkm + 0.1)
        log_subset = np.log10(rpkm_subset + 0.1)
        # Compute z-score stats from the Xpresso gene subset only
        col_mean = log_subset.mean(axis=0)
        col_std = log_subset.std(axis=0)
        col_std = col_std.replace(0, 1)  # avoid division by zero
        # Apply to all genes using subset-derived stats
        expr = (log_all - col_mean) / col_std
        print("  Applied log10(RPKM + 0.1) → z-score per cell type "
              f"(stats from {len(rpkm_subset)} genes) [Xpresso convention]")
    elif args.log_transform == "log2":
        expr = np.log2(rpkm + 1)
        print("  Applied log2(RPKM + 1)")
    else:
        expr = rpkm.copy()
        print("  No transformation applied")

    # Build name mapping: prefer known names, else use EG.name.txt
    eg_map = dict(zip(eg_names["epigenome_id"], eg_names["name"]))
    eg_map.update(_ROADMAP_CELL_NAMES)  # override with cleaner names

    # Rename columns: E003 → Actual_H1, E118 → Actual_K562, etc.
    renamed = {}
    for col in expr.columns:
        cell_name = eg_map.get(col, col)
        renamed[col] = f"Actual_{cell_name}"
    expr_renamed = expr.rename(columns=renamed)

    # Also keep RPKM columns with original Roadmap IDs
    rpkm_renamed = rpkm.rename(columns={c: f"RPKM_{c}" for c in rpkm.columns})

    # ------------------------------------------------------------------
    # 5. Merge with gene info
    # ------------------------------------------------------------------
    result = gene_info[["ENSID", "chrom", "start", "end", "strand",
                        "biotype", "gene_name"]].copy()
    result = result.rename(columns={"gene_name": "Gene name"})
    result = result.merge(rpkm_renamed, left_on="ENSID", right_index=True, how="inner")
    result = result.merge(expr_renamed, left_on="ENSID", right_index=True, how="inner")

    print(f"  {len(result)} genes with Roadmap expression")

    # ------------------------------------------------------------------
    # 6. Merge with Xpresso features (if provided)
    # ------------------------------------------------------------------
    if args.xpresso_csv:
        xpresso_path = os.path.abspath(args.xpresso_csv)
        if not os.path.isfile(xpresso_path):
            print(f"  WARNING: Xpresso CSV not found: {xpresso_path}")
        else:
            print(f"  Merging with Xpresso features from {xpresso_path} ...")
            xp = pd.read_csv(xpresso_path)

            # Feature columns to take from Xpresso
            feature_cols = [
                "ENSID", "TSS_xpresso",
                "UTR5LEN_log10zscore", "CDSLEN_log10zscore",
                "INTRONLEN_log10zscore", "UTR3LEN_log10zscore",
                "UTR5GC", "CDSGC", "UTR3GC", "ORFEXONDENSITY",
            ]
            # Only keep columns that exist
            feature_cols = [c for c in feature_cols if c in xp.columns]
            xp_feats = xp[feature_cols].drop_duplicates(subset=["ENSID"])

            n_before = len(result)
            result = result.merge(xp_feats, on="ENSID", how="inner")
            print(f"  {len(result)}/{n_before} genes matched Xpresso features")

            # Expression targets are pure Roadmap-derived (no Xpresso overwrite)

    # ------------------------------------------------------------------
    # 7. Write final output
    # ------------------------------------------------------------------
    # Add gene_id column (alias of ENSID, used by train_EPInformer_abc.py)
    result.insert(0, "gene_id", result["ENSID"])

    out_path = os.path.join(out_dir, "roadmap_expression_all.csv")
    result.to_csv(out_path, index=False)
    print(f"\nDone! {len(result)} genes × {len(result.columns)} columns")
    print(f"  → {out_path}")

    # Print available Actual_* columns
    actual_cols = sorted([c for c in result.columns if c.startswith("Actual_")])
    print(f"\n{len(actual_cols)} expression targets available:")
    for c in actual_cols:
        print(f"  {c}")


if __name__ == "__main__":
    main()
