"""
Streamlined ABC (Activity-by-Contact) pipeline for EPInformer.

Produces EnhancerPredictionsAllPutative.txt, EnhancerList.txt, GeneList.txt,
and sequence encoder pre-training data from BAM/BigWig inputs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from .utils import StepLogger, check_dependencies, ensure_bam_indexed, load_gene_bed
from .candidates import define_candidates, load_candidates_from_peaks
from .neighborhoods import quantify_neighborhoods
from .predictions import predict_abc
from .encoder_data import generate_encoder_data
from .contact import load_hic

# ---------------------------------------------------------------------------
# Cell-type presets
# ---------------------------------------------------------------------------

_REF_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "hg38"

PRESETS = {
    "K562": {
        "gene_bed": str(_REF_DIR / "CollapsedGeneBounds.hg38.bed"),
        "chrom_sizes": str(_REF_DIR / "GRCh38_EBV.chrom.sizes.tsv"),
        "expression": str(_REF_DIR / "K562_expression.tsv"),
        "qnorm_ref": str(_REF_DIR / "EnhancersQNormRef.K562.txt"),
        "roadmap_id": "E118",
    },
    "GM12878": {
        "gene_bed": str(_REF_DIR / "CollapsedGeneBounds.hg38.bed"),
        "chrom_sizes": str(_REF_DIR / "GRCh38_EBV.chrom.sizes.tsv"),
        "expression": str(_REF_DIR / "GM12878_expression.tsv"),
        "qnorm_ref": str(_REF_DIR / "EnhancersQNormRef.K562.txt"),
        "roadmap_id": "E116",
    },
    "H1": {
        "gene_bed": str(_REF_DIR / "CollapsedGeneBounds.hg38.bed"),
        "chrom_sizes": str(_REF_DIR / "GRCh38_EBV.chrom.sizes.tsv"),
        "expression": str(_REF_DIR / "H1_expression.tsv"),
        "qnorm_ref": str(_REF_DIR / "EnhancersQNormRef.K562.txt"),
        "roadmap_id": "E003",
    },
    "HUVEC": {
        "gene_bed": str(_REF_DIR / "CollapsedGeneBounds.hg38.bed"),
        "chrom_sizes": str(_REF_DIR / "GRCh38_EBV.chrom.sizes.tsv"),
        "expression": str(_REF_DIR / "HUVEC_expression.tsv"),
        "qnorm_ref": str(_REF_DIR / "EnhancersQNormRef.K562.txt"),
        "roadmap_id": "E122",
    },
    "NHEK": {
        "gene_bed": str(_REF_DIR / "CollapsedGeneBounds.hg38.bed"),
        "chrom_sizes": str(_REF_DIR / "GRCh38_EBV.chrom.sizes.tsv"),
        "expression": str(_REF_DIR / "NHEK_expression.tsv"),
        "qnorm_ref": str(_REF_DIR / "EnhancersQNormRef.K562.txt"),
        "roadmap_id": "E127",
    },
}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_abc_pipeline(
    accessibility_bam: str,
    output_dir: str,
    *,
    assay: str = "dnase",
    h3k27ac_bam: Optional[str] = None,
    hic_file: Optional[str] = None,
    gene_bed: Optional[str] = None,
    chrom_sizes: Optional[str] = None,
    expression: Optional[str] = None,
    expression_column: Optional[str] = None,
    fasta: Optional[str] = None,
    qnorm_ref: Optional[str] = None,
    cell_type: str = "K562",
    preset: Optional[str] = None,
    peaks_file: Optional[str] = None,
    n_top_peaks: int = 150_000,
    peak_extend: int = 250,
    window: int = 5_000_000,
    gamma: float = 0.87,
    tss_slop: int = 500,
    hic_resolution: int = 5000,
    blacklist: Optional[str] = None,
    neg_fraction: float = 0.05,
    max_encoder_peaks: int = 100_000,
    include_self_promoter: bool = False,
    dry_run: bool = False,
    n_threads: int = 1,
) -> dict:
    """Run the full ABC pipeline.

    Returns a dict of output file paths.
    """
    # ---- Resolve preset defaults ----
    if preset and preset in PRESETS:
        p = PRESETS[preset]
        gene_bed = gene_bed or p.get("gene_bed")
        chrom_sizes = chrom_sizes or p.get("chrom_sizes")
        expression = expression or p.get("expression")
        qnorm_ref = qnorm_ref or p.get("qnorm_ref")
        expression_column = expression_column or p.get("roadmap_id")

    # ---- Resolve fasta default ----
    if fasta is None:
        default_fasta = Path(__file__).resolve().parent.parent.parent / "data_EPInformer" / "hg38.fa"
        if default_fasta.exists():
            fasta = str(default_fasta)

    os.makedirs(output_dir, exist_ok=True)

    # ---- Print banner ----
    print("=" * 80)
    print("ABC Pipeline for EPInformer")
    print("=" * 80)
    print(f"  Accessibility BAM: {accessibility_bam}")
    print(f"  H3K27ac BAM:       {h3k27ac_bam or '(none, DNase-only activity)'}")
    print(f"  Hi-C:              {hic_file or '(none, power-law fallback)'}")
    print(f"  Gene list:         {gene_bed or '(not provided)'}")
    print(f"  Expression:        {expression or '(not provided)'}")
    print(f"  Cell type:         {cell_type}")
    print(f"  Assay:             {assay}")
    print(f"  Threads:           {n_threads}")
    print(f"  Output:            {output_dir}")
    print("=" * 80)

    if dry_run:
        _dry_run_validate(
            accessibility_bam, h3k27ac_bam, hic_file, gene_bed,
            expression, chrom_sizes, fasta, output_dir,
        )
        return {}

    # ---- Check dependencies ----
    check_dependencies(require_hic=hic_file is not None)

    # ---- Ensure BAMs are sorted and indexed ----
    print("Checking BAM files ...")
    ensure_bam_indexed(accessibility_bam)
    if h3k27ac_bam:
        ensure_bam_indexed(h3k27ac_bam)

    # ---- Determine steps ----
    total_steps = 4  # candidates, neighborhoods, predictions, encoder_data
    logger = StepLogger(total_steps)

    outputs = {}

    # ---- Step 1: Candidate elements ----
    logger.start_step("Defining candidate elements")
    if peaks_file:
        candidates_bed = load_candidates_from_peaks(
            peaks_file, output_dir, logger,
            n_top_peaks=n_top_peaks, peak_extend=peak_extend, blacklist=blacklist,
        )
    else:
        candidates_bed = define_candidates(
            accessibility_bam, output_dir, logger,
            assay=assay, n_top_peaks=n_top_peaks,
            peak_extend=peak_extend, blacklist=blacklist,
        )
    outputs["candidates_bed"] = candidates_bed
    logger.done()

    # ---- Step 2: Neighborhoods ----
    logger.start_step("Quantifying enhancer activity (neighborhoods)")
    gene_df = load_gene_bed(
        gene_bed,
        expression_csv=expression,
        expression_column=expression_column,
    )
    enhancer_list_path, gene_list_path = quantify_neighborhoods(
        candidates_bed, gene_df,
        accessibility_bam=accessibility_bam,
        h3k27ac_bam=h3k27ac_bam,
        output_dir=output_dir,
        logger=logger,
        tss_slop=tss_slop,
        qnorm_ref=qnorm_ref,
        n_threads=n_threads,
    )
    outputs["enhancer_list"] = enhancer_list_path
    outputs["gene_list"] = gene_list_path
    logger.done()

    # ---- Step 3: ABC predictions ----
    logger.start_step("Computing ABC scores")
    predictions_path = predict_abc(
        enhancer_list_path, gene_list_path, output_dir, logger,
        hic_file=hic_file, window=window, gamma=gamma,
        tss_slop=tss_slop, hic_resolution=hic_resolution,
        cell_type=cell_type, n_threads=n_threads,
    )
    outputs["predictions"] = predictions_path
    logger.done()

    # ---- Step 4: Sequence encoder training data ----
    logger.start_step("Generating sequence encoder training data")
    summits_bed = os.path.join(output_dir, "candidates_with_summits.bed")
    if not os.path.exists(summits_bed):
        logger.info("Warning: candidates_with_summits.bed not found, skipping encoder data")
        encoder_csv = None
    elif fasta is None:
        logger.info("Warning: no FASTA provided, skipping encoder data")
        encoder_csv = None
    elif chrom_sizes is None:
        logger.info("Warning: no chrom sizes provided, skipping encoder data")
        encoder_csv = None
    else:
        encoder_csv = generate_encoder_data(
            enhancer_list_path, summits_bed, fasta, chrom_sizes,
            output_dir, logger,
            cell_type=cell_type, neg_fraction=neg_fraction,
            blacklist=blacklist, n_threads=n_threads,
            accessibility_bam=accessibility_bam,
            h3k27ac_bam=h3k27ac_bam,
            max_peaks=max_encoder_peaks,
        )
    outputs["encoder_data"] = encoder_csv
    logger.done()

    # ---- Summary ----
    logger.summary(outputs)
    return outputs


def _dry_run_validate(accessibility_bam, h3k27ac_bam, hic_file, gene_bed,
                      expression, chrom_sizes, fasta, output_dir):
    """Validate inputs without running anything."""
    import os
    print("\n[Dry Run] Validating inputs ...")
    all_ok = True
    for label, path in [
        ("Accessibility BAM", accessibility_bam),
        ("H3K27ac BAM", h3k27ac_bam),
        ("Hi-C", hic_file),
        ("Gene list", gene_bed),
        ("Expression", expression),
        ("Chrom sizes", chrom_sizes),
        ("FASTA", fasta),
    ]:
        if path is None:
            status = "(not provided)"
        elif os.path.isfile(path):
            size_mb = os.path.getsize(path) / 1e6
            status = f"OK ({size_mb:.1f} MB)"
        else:
            status = "MISSING"
            all_ok = False
        print(f"  {label:20s} {path or '-':50s} {status}")

    if os.path.isdir(output_dir):
        print(f"  {'Output dir':20s} {output_dir:50s} (exists)")
    else:
        print(f"  {'Output dir':20s} {output_dir:50s} (will be created)")

    if all_ok:
        print("\n[Dry Run] No issues found. Remove --dry-run to execute.")
    else:
        print("\n[Dry Run] Some files are missing. Check paths above.")
