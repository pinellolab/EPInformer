#!/usr/bin/env python
"""Build the gene-level EPInformer HDF5 (``{cell}_samples.h5``) for a cell line
from PRECOMPUTED ABC nomination outputs, plus hg38.fa and the gene-expression CSV.

This is the encoding stage (``obtain_PE_withSignals``) pointed directly at an
existing pair of ABC tables instead of re-running Stage-1 links:

  * predictions   ``EnhancerPredictionsAllPutative.txt[.gz]`` — the enhancer->gene
                  links with ``activity_base``, ``hic_contact``, ``distance``,
                  ``TargetGene``/``TargetGeneTSS``, ``isSelfPromoter``.
  * enhancer_list ``EnhancerList.txt`` — per-candidate ``normalized_dhs`` /
                  ``normalized_h3K27ac`` (merged on the ``name`` column).

Use it to extend expression prediction to cell lines whose raw BAMs are not on
disk but whose ABC links were already computed (e.g. BSCC's
``epinformer_data_20250503`` for H1/HepG2/HUVEC/NHEK, contact = ABC average Hi-C).
No BAM re-counting happens: ``activity_base`` / ``hic_contact`` are read straight
from the ABC tables, so the expression-model ``activity`` feature is
BAM-consistent with the pretrained per-cell encoder (both derive from the same
ABC quantification). CAGE labels do not exist for these cells -> train RNA only.

Example:
  python scripts/build_gene_h5_for_cell.py \
    --cell H1 \
    --predictions   .../DNase_ENCFF761ZR_avghic_noCutOff_1MB_predictions/EnhancerPredictionsAllPutative.txt.gz \
    --enhancer_list .../DNase_ENCFF761ZRE_Neighborhoods/EnhancerList.txt \
    --fasta    data/reference/hg38/hg38.fa \
    --expr_csv data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv \
    --output_dir batch_output/H1/encoding \
    --max_distance 100000 --n_enhancer 60 --max_seq_len 2000 --include_self_promoter
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.pipelines_legacy import obtain_PE_withSignals


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--predictions", required=True,
                    help="EnhancerPredictionsAllPutative.txt[.gz]")
    ap.add_argument("--enhancer_list", required=True,
                    help="Neighborhoods/EnhancerList.txt")
    ap.add_argument("--fasta", default="data/reference/hg38/hg38.fa")
    ap.add_argument("--expr_csv",
                    default="data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_distance", type=int, default=100_000)
    ap.add_argument("--min_distance", type=int, default=0)
    ap.add_argument("--n_enhancer", type=int, default=60)
    ap.add_argument("--max_seq_len", type=int, default=2000)
    ap.add_argument("--include_self_promoter", action="store_true")
    args = ap.parse_args()

    for p in (args.predictions, args.enhancer_list, args.fasta, args.expr_csv):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing input file: {p}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[build_gene_h5] cell={args.cell}")
    print(f"  predictions   = {args.predictions}")
    print(f"  enhancer_list = {args.enhancer_list}")
    print(f"  fasta         = {args.fasta}")
    print(f"  expr_csv      = {args.expr_csv}")
    print(f"  output_dir    = {args.output_dir}")
    print(f"  max_distance={args.max_distance} n_enhancer={args.n_enhancer} "
          f"max_seq_len={args.max_seq_len} include_self_promoter={args.include_self_promoter}")

    obtain_PE_withSignals(
        fname=[args.predictions, args.enhancer_list],
        max_distance=args.max_distance,
        min_distance=args.min_distance,
        n_enhancer=args.n_enhancer,
        max_seq_len=args.max_seq_len,
        gene_expression_csv=args.expr_csv,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        include_self_promoter=args.include_self_promoter,
        cell_type=args.cell,
        h5_name=f"{args.cell}_samples.h5",
    )
    out_h5 = os.path.join(args.output_dir, f"{args.cell}_samples.h5")
    print(f"DONE: {out_h5}  exists={os.path.exists(out_h5)}")


if __name__ == "__main__":
    main()
