"""
Quantify enhancer activity and classify candidate elements (Step 2).

Counts accessibility and H3K27ac reads in candidate regions and gene
promoters, applies optional quantile normalization, classifies elements
as promoter/genic/intergenic, and computes activity scores following
the ABC model framework.
"""

import os

import numpy as np
import pandas as pd
import pybedtools
from scipy.stats import rankdata
from tqdm import tqdm

from .utils import count_reads_in_regions, quantile_normalize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_candidates(candidates_bed: str) -> pd.DataFrame:
    """Load a 3-column BED file of candidate elements."""
    df = pd.read_csv(
        candidates_bed,
        sep="\t",
        header=None,
        names=["chr", "start", "end"],
        comment="#",
    )
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


def _count_signal(bam_path: str, regions: pd.DataFrame, prefix: str, n_threads: int = 1) -> pd.DataFrame:
    """Count reads and rename columns with *prefix* (e.g. 'DHS' or 'H3K27ac')."""
    result = count_reads_in_regions(bam_path, regions, n_threads=n_threads)
    rename_map = {
        "readCount": f"{prefix}.readCount",
        "RPM": f"{prefix}.RPM",
        "RPKM": f"{prefix}.RPKM",
    }
    result = result.rename(columns=rename_map)
    return result


def _apply_qnorm(df: pd.DataFrame, qnorm_ref: str, has_h3k27ac: bool) -> pd.DataFrame:
    """Apply quantile normalization to RPM columns using a reference file."""
    ref_df = pd.read_csv(qnorm_ref, sep="\t")

    dhs_ref = np.sort(ref_df["DHS.RPM"].values)
    df["DHS.RPM.quantile"] = quantile_normalize(df["DHS.RPM"].values, reference=dhs_ref)

    if has_h3k27ac and "H3K27ac.RPM" in ref_df.columns:
        h3k_ref = np.sort(ref_df["H3K27ac.RPM"].values)
        df["H3K27ac.RPM.quantile"] = quantile_normalize(
            df["H3K27ac.RPM"].values, reference=h3k_ref
        )

    return df


def _classify_candidates(
    candidates: pd.DataFrame,
    gene_bed_df: pd.DataFrame,
    tss_slop: int,
) -> pd.DataFrame:
    """Classify candidates as promoter, genic, or intergenic.

    Uses pybedtools intersect to determine overlap with promoter regions
    (TSS +/- tss_slop) and gene bodies.
    """
    cand_bed = pybedtools.BedTool.from_dataframe(
        candidates[["chr", "start", "end"]]
    )

    # --- Promoter regions: TSS +/- tss_slop ---
    promoter_regions = pd.DataFrame({
        "chr": gene_bed_df["chr"],
        "start": (gene_bed_df["tss"] - tss_slop).clip(lower=0),
        "end": gene_bed_df["tss"] + tss_slop,
    })
    promoter_bed = pybedtools.BedTool.from_dataframe(
        promoter_regions[["chr", "start", "end"]]
    ).sort()

    # --- Gene body regions ---
    gene_body = gene_bed_df[["chr", "start", "end"]].copy()
    gene_body_bed = pybedtools.BedTool.from_dataframe(
        gene_body[["chr", "start", "end"]]
    ).sort()

    # Intersect candidates with promoter regions
    promoter_hits = cand_bed.intersect(promoter_bed, u=True, sorted=False)
    promoter_set = set()
    for interval in promoter_hits:
        promoter_set.add((str(interval.chrom), int(interval.start), int(interval.end)))

    # Intersect candidates with gene bodies
    genic_hits = cand_bed.intersect(gene_body_bed, u=True, sorted=False)
    genic_set = set()
    for interval in genic_hits:
        genic_set.add((str(interval.chrom), int(interval.start), int(interval.end)))

    # Classify each candidate
    classes = []
    for _, row in tqdm(candidates.iterrows(), total=len(candidates),
                       desc="  Classifying elements", leave=False):
        key = (str(row["chr"]), int(row["start"]), int(row["end"]))
        if key in promoter_set:
            classes.append("promoter")
        elif key in genic_set:
            classes.append("genic")
        else:
            classes.append("intergenic")

    candidates = candidates.copy()
    candidates["class"] = classes
    candidates["name"] = (
        candidates["class"] + "|"
        + candidates["chr"].astype(str) + ":"
        + candidates["start"].astype(str) + "-"
        + candidates["end"].astype(str)
    )

    pybedtools.cleanup()
    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantify_neighborhoods(
    candidates_bed: str,
    gene_bed_df: pd.DataFrame,
    accessibility_bam: str,
    h3k27ac_bam: str = None,
    output_dir: str = ".",
    logger=None,
    tss_slop: int = 500,
    qnorm_ref: str = None,
    n_threads: int = 1,
) -> tuple:
    """Quantify enhancer activity for candidate elements (ABC Step 2).

    Counts reads in candidate regions and gene promoters, optionally
    applies quantile normalization, classifies elements, computes
    activity scores, and writes EnhancerList.txt and GeneList.txt.

    Parameters
    ----------
    candidates_bed : str
        Path to candidates.bed (chr, start, end).
    gene_bed_df : pandas.DataFrame
        DataFrame from ``load_gene_bed()`` with columns: chr, start, end,
        symbol, ENSID, strand, gene_type, tss, Expression.
    accessibility_bam : str
        Path to DNase/ATAC BAM file.
    h3k27ac_bam : str, optional
        Path to H3K27ac BAM file.
    output_dir : str
        Output directory (default ".").
    logger : StepLogger, optional
        Logger instance for progress messages.
    tss_slop : int
        Bases around TSS for promoter definition (default 500).
    qnorm_ref : str, optional
        Path to quantile normalization reference file (one value per line).

    Returns
    -------
    tuple of (str, str)
        Paths to (EnhancerList.txt, GeneList.txt).
    """
    os.makedirs(output_dir, exist_ok=True)
    has_h3k27ac = h3k27ac_bam is not None

    # Helper for optional logging
    def _log(msg):
        if logger is not None:
            logger.info(msg)

    # ------------------------------------------------------------------
    # 1. Load candidates
    # ------------------------------------------------------------------
    candidates = _load_candidates(candidates_bed)
    n_cand = len(candidates)

    # ------------------------------------------------------------------
    # 2. Count reads in candidates
    # ------------------------------------------------------------------
    _log(f"Counting DNase reads in {n_cand} candidates (threads={n_threads}) ...")
    candidates = _count_signal(accessibility_bam, candidates, "DHS", n_threads=n_threads)

    if has_h3k27ac:
        _log(f"Counting H3K27ac reads in {n_cand} candidates (threads={n_threads}) ...")
        candidates = _count_signal(h3k27ac_bam, candidates, "H3K27ac", n_threads=n_threads)

    # ------------------------------------------------------------------
    # 3. Quantile normalization
    # ------------------------------------------------------------------
    if qnorm_ref is not None:
        _log(f"Applying quantile normalization from {qnorm_ref} ...")
        candidates = _apply_qnorm(candidates, qnorm_ref, has_h3k27ac)
    else:
        candidates["DHS.RPM.quantile"] = candidates["DHS.RPM"].copy()
        if has_h3k27ac:
            candidates["H3K27ac.RPM.quantile"] = candidates["H3K27ac.RPM"].copy()

    # ------------------------------------------------------------------
    # 4. Count reads in gene promoter regions (TSS +/- 1kb)
    # ------------------------------------------------------------------
    _log("Counting reads in gene promoter regions (TSS +/- 1kb) ...")
    gene_df = gene_bed_df.copy()

    promoter_df = pd.DataFrame({
        "chr": gene_df["chr"],
        "start": (gene_df["tss"] - 1000).clip(lower=0),
        "end": gene_df["tss"] + 1000,
    })

    promoter_counts = count_reads_in_regions(accessibility_bam, promoter_df, n_threads=n_threads)
    gene_df["DHS.RPM.TSS1Kb"] = promoter_counts["RPM"].values

    if has_h3k27ac:
        promoter_counts_h3k = count_reads_in_regions(h3k27ac_bam, promoter_df, n_threads=n_threads)
        gene_df["H3K27ac.RPM.TSS1Kb"] = promoter_counts_h3k["RPM"].values
        gene_df["PromoterActivity"] = np.sqrt(
            gene_df["H3K27ac.RPM.TSS1Kb"] * gene_df["DHS.RPM.TSS1Kb"]
        )
    else:
        gene_df["H3K27ac.RPM.TSS1Kb"] = 0.0
        gene_df["PromoterActivity"] = gene_df["DHS.RPM.TSS1Kb"]

    # Rank-based quantile (0 to 1)
    n_genes = len(gene_df)
    if n_genes > 0:
        gene_df["PromoterActivityQuantile"] = (
            rankdata(gene_df["PromoterActivity"], method="average") / n_genes
        )
    else:
        gene_df["PromoterActivityQuantile"] = np.nan

    # ------------------------------------------------------------------
    # 5. Classify candidates
    # ------------------------------------------------------------------
    _log("Classifying candidate elements ...")
    candidates = _classify_candidates(candidates, gene_bed_df, tss_slop)

    n_promoter = (candidates["class"] == "promoter").sum()
    n_genic = (candidates["class"] == "genic").sum()
    n_intergenic = (candidates["class"] == "intergenic").sum()
    _log(
        f"Classifying elements: {n_promoter} promoter | "
        f"{n_genic} genic | {n_intergenic} intergenic"
    )

    # ------------------------------------------------------------------
    # 6. Compute activity scores
    # ------------------------------------------------------------------
    if has_h3k27ac:
        candidates["activity_base"] = np.sqrt(
            candidates["H3K27ac.RPM.quantile"] * candidates["DHS.RPM.quantile"]
        )
        h3k27ac_sum = candidates["H3K27ac.RPM"].sum()
        candidates["normalized_h3K27ac"] = (
            candidates["H3K27ac.RPM"] / h3k27ac_sum if h3k27ac_sum > 0 else 0.0
        )
    else:
        candidates["activity_base"] = candidates["DHS.RPM.quantile"]
        candidates["normalized_h3K27ac"] = 0.0

    dhs_sum = candidates["DHS.RPM"].sum()
    candidates["normalized_dhs"] = (
        candidates["DHS.RPM"] / dhs_sum if dhs_sum > 0 else 0.0
    )

    # Boolean classification columns
    candidates["isPromoterElement"] = (candidates["class"] == "promoter").astype(int)
    candidates["isGenicElement"] = (candidates["class"] == "genic").astype(int)
    candidates["isIntergenicElement"] = (candidates["class"] == "intergenic").astype(int)

    # ------------------------------------------------------------------
    # 7. Write EnhancerList.txt
    # ------------------------------------------------------------------
    enhancer_cols = [
        "chr", "start", "end", "name", "class",
        "DHS.RPM", "DHS.RPM.quantile",
    ]
    if has_h3k27ac:
        enhancer_cols += ["H3K27ac.RPM", "H3K27ac.RPM.quantile"]
    else:
        candidates["H3K27ac.RPM"] = 0.0
        candidates["H3K27ac.RPM.quantile"] = 0.0
        enhancer_cols += ["H3K27ac.RPM", "H3K27ac.RPM.quantile"]

    enhancer_cols += [
        "activity_base", "normalized_h3K27ac", "normalized_dhs",
        "isPromoterElement", "isGenicElement", "isIntergenicElement",
    ]

    enhancer_list_path = os.path.join(output_dir, "EnhancerList.txt")
    candidates[enhancer_cols].to_csv(
        enhancer_list_path, sep="\t", index=False
    )
    _log(f"Saved: {enhancer_list_path}")

    # ------------------------------------------------------------------
    # 8. Write GeneList.txt
    # ------------------------------------------------------------------
    gene_out = pd.DataFrame({
        "chr": gene_df["chr"],
        "start": gene_df["start"],
        "end": gene_df["end"],
        "name": gene_df["ENSID"],
        "score": 0,
        "strand": gene_df["strand"],
        "Ensembl_ID": gene_df["ENSID"],
        "gene_type": gene_df["gene_type"],
        "symbol": gene_df["symbol"],
        "tss": gene_df["tss"],
        "Expression": gene_df.get("Expression", np.nan),
        "PromoterActivityQuantile": gene_df["PromoterActivityQuantile"],
        "DHS.RPM.TSS1Kb": gene_df["DHS.RPM.TSS1Kb"],
        "H3K27ac.RPM.TSS1Kb": gene_df["H3K27ac.RPM.TSS1Kb"],
    })

    gene_list_path = os.path.join(output_dir, "GeneList.txt")
    gene_out.to_csv(gene_list_path, sep="\t", index=False)
    _log(f"Saved: {gene_list_path}")

    return (enhancer_list_path, gene_list_path)
