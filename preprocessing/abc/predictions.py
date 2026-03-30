"""
Compute ABC scores for gene-enhancer pairs (Step 3).

Pairs each gene with candidate enhancers within max_distance of TSS, computes
contact scores (power-law and optional Hi-C), calculates ABC scores, and
writes the final EnhancerPredictionsAllPutative.txt output.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from .contact import load_hic, get_contacts_for_pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _pair_chromosome(chrom, enh_group, gene_group, max_distance):
    """Build gene-enhancer pairs for a single chromosome (process worker)."""
    enh_sub = enh_group.copy()
    gene_sub = gene_group.copy()
    enh_sub["_merge_key"] = 1
    gene_sub["_merge_key"] = 1
    cross = enh_sub.merge(gene_sub, on="_merge_key", suffixes=("", "_gene"))
    cross.drop(columns=["_merge_key"], inplace=True)
    cross["distance"] = (cross["element_mid"] - cross["tss"]).abs()
    cross = cross[cross["distance"] <= max_distance].copy()
    return cross


def predict_abc(
    enhancer_list_path: str,
    gene_list_path: str,
    output_dir: str,
    logger,
    hic_file: str = None,
    max_distance: int = 2_500_000,
    gamma: float = 0.87,
    tss_slop: int = 500,
    hic_resolution: int = 5000,
    cell_type: str = "K562",
    n_threads: int = 1,
) -> str:
    """Compute ABC scores for all gene-enhancer pairs (ABC Step 3).

    Parameters
    ----------
    enhancer_list_path : str
        Path to EnhancerList.txt from Step 2.
    gene_list_path : str
        Path to GeneList.txt from Step 2.
    output_dir : str
        Directory where output will be written.
    logger : StepLogger
        Logger instance for progress messages.
    hic_file : str, optional
        Path to a ``.hic`` file for Hi-C contact lookup.
    max_distance : int
        Maximum distance (bp) for gene-enhancer pairing (default 5 Mb).
    gamma : float
        Power-law exponent for contact estimation (default 0.87).
    tss_slop : int
        TSS +/- this distance defines the self-promoter region (default 500).
    hic_resolution : int
        Hi-C bin resolution in bp (default 5000).
    cell_type : str
        Label for the CellType column in the output (default ``"K562"``).

    Returns
    -------
    str
        Path to the written EnhancerPredictionsAllPutative.txt file.
    """

    # Helper for logging
    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    # ------------------------------------------------------------------
    # 1. Load EnhancerList.txt and GeneList.txt
    # ------------------------------------------------------------------
    enhancers = pd.read_csv(enhancer_list_path, sep="\t")
    genes = pd.read_csv(gene_list_path, sep="\t")

    # Compute element midpoints for enhancers
    enhancers["element_mid"] = ((enhancers["start"] + enhancers["end"]) / 2).astype(int)

    # ------------------------------------------------------------------
    # 2. Optionally load Hi-C data
    # ------------------------------------------------------------------
    hic_data = None
    if hic_file is not None:
        _log(f"Loading Hi-C data from {hic_file} ...")
        hic_data = load_hic(hic_file, resolution=hic_resolution)

    # ------------------------------------------------------------------
    # 3. Build gene-enhancer pairs within max_distance of TSS
    # ------------------------------------------------------------------
    _log(f"Building gene-enhancer pairs (max_distance={max_distance / 1e6:.0f}Mb) ...")

    # Build per-chromosome pairing tasks
    enh_groups = {chrom: grp for chrom, grp in enhancers.groupby("chr")}
    gene_groups = {chrom: grp for chrom, grp in genes.groupby("chr")}
    chrom_list = [c for c in enh_groups if c in gene_groups]

    pair_frames = []
    if n_threads <= 1 or len(chrom_list) <= 1:
        # Sequential path
        for chrom in tqdm(chrom_list, desc="  Pairing chromosomes", leave=False, ncols=80):
            cross = _pair_chromosome(chrom, enh_groups[chrom], gene_groups[chrom], max_distance)
            if not cross.empty:
                pair_frames.append(cross)
    else:
        # Parallel: one process per chromosome
        with ProcessPoolExecutor(max_workers=min(n_threads, len(chrom_list))) as pool:
            futures = {
                chrom: pool.submit(_pair_chromosome, chrom, enh_groups[chrom], gene_groups[chrom], max_distance)
                for chrom in chrom_list
            }
            for chrom, future in tqdm(futures.items(), total=len(futures),
                                      desc="  Pairing chromosomes", leave=False, ncols=80):
                cross = future.result()
                if not cross.empty:
                    pair_frames.append(cross)

    if not pair_frames:
        _log(f"WARNING: No gene-enhancer pairs found within {max_distance / 1e6:.1f}Mb of TSS.")
        # Write an empty output file
        pred_dir = os.path.join(output_dir, "Predictions")
        os.makedirs(pred_dir, exist_ok=True)
        out_path = os.path.join(pred_dir, "EnhancerPredictionsAllPutative.txt")
        pd.DataFrame().to_csv(out_path, sep="\t", index=False)
        return out_path

    pairs = pd.concat(pair_frames, ignore_index=True)
    n_genes = pairs["symbol"].nunique() if "symbol" in pairs.columns else pairs["tss"].nunique()
    _log(f"Generated {len(pairs)} pairs across {n_genes} genes")

    # ------------------------------------------------------------------
    # 4. Compute contact scores
    # ------------------------------------------------------------------
    # Prepare columns expected by get_contacts_for_pairs
    pairs["chrom"] = pairs["chr"]

    pairs = get_contacts_for_pairs(
        pairs, hic_data=hic_data, gamma=gamma, resolution=hic_resolution
    )

    # Determine which contact column to use for ABC scoring
    if hic_data is not None and "hic_contact_pl_scaled" in pairs.columns:
        pairs["contact"] = pairs["hic_contact_pl_scaled"]
    else:
        pairs["contact"] = pairs["powerlaw_contact"]

    # ------------------------------------------------------------------
    # 5. Compute ABC scores (per-gene normalization)
    # ------------------------------------------------------------------
    _log("Computing ABC scores ...")

    # ABC.Score.Numerator = activity_base * contact
    pairs["ABC.Score.Numerator"] = pairs["activity_base"] * pairs["contact"]

    # powerlaw.Score.Numerator = activity_base * powerlaw_contact
    pairs["powerlaw.Score.Numerator"] = pairs["activity_base"] * pairs["powerlaw_contact"]

    # Determine gene grouping key
    # Use (symbol, tss) combination if symbol is available, otherwise tss alone
    if "symbol" in pairs.columns:
        gene_key = ["symbol", "tss"]
    else:
        gene_key = ["tss"]

    # Normalize per gene: ABC.Score = numerator / sum(numerator)
    gene_sum_abc = pairs.groupby(gene_key)["ABC.Score.Numerator"].transform("sum")
    pairs["ABC.Score"] = np.where(
        gene_sum_abc > 0,
        pairs["ABC.Score.Numerator"] / gene_sum_abc,
        0.0,
    )

    gene_sum_pl = pairs.groupby(gene_key)["powerlaw.Score.Numerator"].transform("sum")
    pairs["powerlaw.Score"] = np.where(
        gene_sum_pl > 0,
        pairs["powerlaw.Score.Numerator"] / gene_sum_pl,
        0.0,
    )

    # ------------------------------------------------------------------
    # 6. Mark isSelfPromoter
    # ------------------------------------------------------------------
    element_half_width = ((pairs["end"] - pairs["start"]) / 2).values
    pairs["isSelfPromoter"] = (
        (pairs["distance"].values) < (tss_slop + element_half_width)
    )

    # ------------------------------------------------------------------
    # 7. Format output columns
    # ------------------------------------------------------------------

    # Gene-side columns from the gene list (may have suffixes from the merge)
    # Map TargetGene info
    # Resolve ENSID column (may be 'ENSID', 'Ensembl_ID', or with '_gene' suffix)
    _ensid_col = None
    for _c in ("ENSID", "Ensembl_ID", "ENSID_gene", "Ensembl_ID_gene"):
        if _c in pairs.columns:
            _ensid_col = _c
            break
    pairs["TargetGene"] = pairs[_ensid_col] if _ensid_col else pairs["symbol"]
    pairs["TargetGeneSymbol"] = pairs["symbol"]
    pairs["TargetGeneTSS"] = pairs["tss"]

    # Expression
    expr_col = None
    for candidate in ("Expression", "Expression_gene"):
        if candidate in pairs.columns:
            expr_col = candidate
            break
    if expr_col is not None:
        pairs["TargetGeneExpression"] = pairs[expr_col]
    else:
        pairs["TargetGeneExpression"] = np.nan

    # TargetGeneIsExpressed
    pairs["TargetGeneIsExpressed"] = (
        pairs["TargetGeneExpression"].fillna(0) > 0.5
    )

    # PromoterActivityQuantile
    paq_col = None
    for candidate in ("PromoterActivityQuantile", "PromoterActivityQuantile_gene"):
        if candidate in pairs.columns:
            paq_col = candidate
            break
    pairs["TargetGenePromoterActivityQuantile"] = (
        pairs[paq_col] if paq_col is not None else np.nan
    )

    # Fill in columns with reasonable defaults for those we don't fully compute
    if "hic_contact" not in pairs.columns:
        pairs["hic_contact"] = np.nan
    if "hic_contact_pl_scaled" not in pairs.columns:
        pairs["hic_contact_pl_scaled"] = np.nan

    pairs["powerlaw_contact_reference"] = pairs["powerlaw_contact"]
    pairs["hic_pseudocount"] = 0.0
    pairs["hic_contact_pl_scaled_adj"] = pairs["hic_contact_pl_scaled"]

    # Ensure normalized columns exist
    if "normalized_h3K27ac" not in pairs.columns:
        pairs["normalized_h3K27ac"] = 0.0
    if "normalized_dhs" not in pairs.columns:
        pairs["normalized_dhs"] = 0.0

    pairs["CellType"] = cell_type

    # Define final output column order
    output_columns = [
        "chr", "start", "end", "name", "class",
        "activity_base",
        "TargetGene", "TargetGeneSymbol", "TargetGeneTSS", "TargetGeneExpression",
        "TargetGenePromoterActivityQuantile", "TargetGeneIsExpressed",
        "distance", "isSelfPromoter",
        "powerlaw_contact", "powerlaw_contact_reference",
        "hic_contact", "hic_contact_pl_scaled",
        "hic_pseudocount", "hic_contact_pl_scaled_adj",
        "ABC.Score.Numerator", "ABC.Score",
        "powerlaw.Score.Numerator", "powerlaw.Score",
        "CellType",
        "normalized_h3K27ac", "normalized_dhs",
    ]

    # Only keep columns that exist (in case of unexpected missing columns)
    output_columns = [c for c in output_columns if c in pairs.columns]

    output_df = pairs[output_columns].copy()

    # ------------------------------------------------------------------
    # 8. Write output
    # ------------------------------------------------------------------
    pred_dir = os.path.join(output_dir, "Predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_path = os.path.join(pred_dir, "EnhancerPredictionsAllPutative.txt")

    output_df.to_csv(out_path, sep="\t", index=False)
    _log(f"Saved: {out_path} ({len(output_df)} links)")

    return out_path
