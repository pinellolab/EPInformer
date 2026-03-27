#!/usr/bin/env python3
"""
Streamlined ABC pipeline for EPInformer.

Produces ABC-compatible output files from BAM inputs, ready for EPInformer
preprocessing and training. Designed for any cell type.

Examples::

    # Full pipeline with K562 preset (minimal flags)
    python run_abc_pipeline.py full --preset K562 \\
        --accessibility-bam data/K562/DNase/ENCFF257HEE.bam \\
        --output-dir ./abc_output/K562

    # Full pipeline with all inputs
    python run_abc_pipeline.py full \\
        --accessibility-bam data/K562/DNase/ENCFF257HEE.bam \\
        --h3k27ac-bam data/K562/H3K27ac/ENCFF232RQF.bam \\
        --hic data/K562/HiC/ENCFF621AIY.hic \\
        --cell-type K562 --output-dir ./abc_output/K562

    # ATAC-seq input (auto-adjusts MACS2 parameters)
    python run_abc_pipeline.py full \\
        --accessibility-bam my_atac.bam --assay atac \\
        --h3k27ac-bam my_h3k27ac.bam \\
        --preset K562 --output-dir ./abc_output/

    # From pre-called peaks (skip MACS2)
    python run_abc_pipeline.py from-peaks \\
        --peaks peaks.narrowPeak \\
        --accessibility-bam data/K562/DNase/ENCFF257HEE.bam \\
        --h3k27ac-bam data/K562/H3K27ac/ENCFF232RQF.bam \\
        --output-dir ./abc_output/K562

    # Dry run (validate inputs, show config, no execution)
    python run_abc_pipeline.py full --preset K562 \\
        --accessibility-bam my_dnase.bam \\
        --output-dir ./abc_output/ --dry-run

    # Chain into EPInformer preprocessing
    python run_abc_pipeline.py full --preset K562 \\
        --accessibility-bam data/K562/DNase/ENCFF257HEE.bam \\
        --h3k27ac-bam data/K562/H3K27ac/ENCFF232RQF.bam \\
        --output-dir ./abc_output/K562 \\
        --chain-preprocessing --preprocessing-output-dir ./training_data/k562/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _add_shared(p: argparse.ArgumentParser) -> None:
    """Add flags shared across subcommands."""
    p.add_argument(
        "--accessibility-bam", required=True,
        help="DNase-seq or ATAC-seq BAM file (indexed).",
    )
    p.add_argument(
        "--assay", default="dnase", choices=["dnase", "atac"],
        help="Assay type: dnase or atac (adjusts MACS2 params). Default: dnase.",
    )
    p.add_argument("--h3k27ac-bam", default=None, help="H3K27ac ChIP-seq BAM (optional).")
    p.add_argument("--hic", default=None, help="Hi-C .hic file (optional; power-law fallback if omitted).")
    p.add_argument("--output-dir", required=True, help="Output directory.")
    p.add_argument("--cell-type", default="K562", help="Cell type label. Default: K562.")
    p.add_argument(
        "--preset", default=None,
        choices=["K562", "GM12878", "H1", "HUVEC", "NHEK"],
        help="Cell-type preset (auto-fills gene list, expression, chrom sizes, qnorm ref).",
    )
    p.add_argument("--gene-bed", default=None, help="Gene annotations BED (CollapsedGeneBounds.hg38.bed).")
    p.add_argument("--chrom-sizes", default=None, help="Chromosome sizes TSV.")
    p.add_argument("--expression", default=None, help="Gene expression table (Roadmap RNA-seq RPKM).")
    p.add_argument("--expression-column", default=None, help="Column name in expression table for this cell type.")
    p.add_argument("--fasta", default=None, help="Reference genome FASTA (hg38).")
    p.add_argument("--qnorm-ref", default=None, help="Quantile normalization reference file.")
    p.add_argument("--n-top-peaks", type=int, default=150_000, help="Max peaks from MACS2. Default: 150000.")
    p.add_argument("--peak-extend", type=int, default=250, help="Half-width for peak resizing. Default: 250 (=500bp).")
    p.add_argument("--window", type=int, default=5_000_000, help="Max distance for gene-enhancer pairing. Default: 5Mb.")
    p.add_argument("--gamma", type=float, default=0.87, help="Power-law exponent. Default: 0.87.")
    p.add_argument("--tss-slop", type=int, default=500, help="TSS ± this = promoter region. Default: 500.")
    p.add_argument("--hic-resolution", type=int, default=5000, help="Hi-C bin resolution. Default: 5000.")
    p.add_argument("--blacklist", default=None, help="Blacklisted regions BED to exclude.")
    p.add_argument("--neg-fraction", type=float, default=0.05, help="Negative sample fraction for encoder data. Default: 0.05.")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs only, no execution.")
    p.add_argument(
        "--chain-preprocessing", action="store_true",
        help="Auto-run EPInformer preprocessing after ABC pipeline.",
    )
    p.add_argument(
        "--preprocessing-output-dir", default=None,
        help="Output dir for EPInformer preprocessing (used with --chain-preprocessing).",
    )


def cmd_full(args: argparse.Namespace) -> None:
    """Full ABC pipeline from BAM files."""
    from epinformer_preprocessing.abc import run_abc_pipeline

    outputs = run_abc_pipeline(
        accessibility_bam=args.accessibility_bam,
        output_dir=args.output_dir,
        assay=args.assay,
        h3k27ac_bam=args.h3k27ac_bam,
        hic_file=args.hic,
        gene_bed=args.gene_bed,
        chrom_sizes=args.chrom_sizes,
        expression=args.expression,
        expression_column=args.expression_column,
        fasta=args.fasta,
        qnorm_ref=args.qnorm_ref,
        cell_type=args.cell_type,
        preset=args.preset,
        n_top_peaks=args.n_top_peaks,
        peak_extend=args.peak_extend,
        window=args.window,
        gamma=args.gamma,
        tss_slop=args.tss_slop,
        hic_resolution=args.hic_resolution,
        blacklist=args.blacklist,
        neg_fraction=args.neg_fraction,
        dry_run=args.dry_run,
    )

    if args.chain_preprocessing and outputs:
        _chain_preprocessing(args, outputs)


def cmd_from_peaks(args: argparse.Namespace) -> None:
    """ABC pipeline from pre-called peaks (skip MACS2)."""
    from epinformer_preprocessing.abc import run_abc_pipeline

    outputs = run_abc_pipeline(
        accessibility_bam=args.accessibility_bam,
        output_dir=args.output_dir,
        assay=args.assay,
        h3k27ac_bam=args.h3k27ac_bam,
        hic_file=args.hic,
        gene_bed=args.gene_bed,
        chrom_sizes=args.chrom_sizes,
        expression=args.expression,
        expression_column=args.expression_column,
        fasta=args.fasta,
        qnorm_ref=args.qnorm_ref,
        cell_type=args.cell_type,
        preset=args.preset,
        peaks_file=args.peaks,
        n_top_peaks=args.n_top_peaks,
        peak_extend=args.peak_extend,
        window=args.window,
        gamma=args.gamma,
        tss_slop=args.tss_slop,
        hic_resolution=args.hic_resolution,
        blacklist=args.blacklist,
        neg_fraction=args.neg_fraction,
        dry_run=args.dry_run,
    )

    if args.chain_preprocessing and outputs:
        _chain_preprocessing(args, outputs)


def _chain_preprocessing(args, outputs):
    """Chain into EPInformer preprocessing after ABC pipeline."""
    pred_path = outputs.get("predictions")
    enh_path = outputs.get("enhancer_list")
    if not pred_path or not enh_path:
        print("Warning: Cannot chain preprocessing — missing ABC output files.")
        return

    prep_out = args.preprocessing_output_dir or os.path.join(args.output_dir, "preprocessing")
    os.makedirs(prep_out, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("Chaining into EPInformer preprocessing ...")
    print(f"  Predictions:  {pred_path}")
    print(f"  EnhancerList: {enh_path}")
    print(f"  Output:       {prep_out}")
    print(f"{'=' * 80}")

    from epinformer_preprocessing.pipelines_legacy import obtain_PE

    fasta = args.fasta
    if fasta is None:
        default_fasta = Path(__file__).resolve().parent / "data_EPInformer" / "hg38.fa"
        if default_fasta.exists():
            fasta = str(default_fasta)

    gene_expr = args.expression
    if gene_expr is None and args.preset:
        from epinformer_preprocessing.abc import PRESETS
        gene_expr = PRESETS.get(args.preset, {}).get("expression")

    obtain_PE(
        [pred_path, enh_path],
        [],  # no BigWig signals in chain mode
        max_distance=args.window,
        add_flank=False,
        use_strand=False,
        n_enhancer=60,
        max_seq_len=2000,
        pe_type="AllPutative",
        cell_type=args.cell_type,
        gene_expression_csv=gene_expr,
        fasta_path=fasta,
        output_dir=prep_out,
    )
    print(f"EPInformer preprocessing complete → {prep_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streamlined ABC pipeline for EPInformer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- full ----
    p_full = sub.add_parser("full", help="Full ABC pipeline from BAM files")
    _add_shared(p_full)
    p_full.set_defaults(func=cmd_full)

    # ---- from-peaks ----
    p_peaks = sub.add_parser("from-peaks", help="ABC pipeline from pre-called peaks (skip MACS2)")
    _add_shared(p_peaks)
    p_peaks.add_argument("--peaks", required=True, help="Pre-called peaks file (narrowPeak format).")
    p_peaks.set_defaults(func=cmd_from_peaks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
