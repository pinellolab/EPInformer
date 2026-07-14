#!/usr/bin/env python
"""Regenerate ONLY the encoder-pretrain data (ABC Step 4) from the ENCODE
H3K27ac narrowPeak, without redoing ABC Steps 1-3.

Uses ``encoder_peaks_file`` from config/samples.tsv (the ENCODE H3K27ac
narrowPeak) as the peak/summit source and the current 3-window / 100bp-overlap
settings in preprocessing/abc/encoder_pretrain_data.py. Writes
``batch_output/{cell}/links/{cell}_peak_3bins_around_summit_activity_sequence.csv``.

Usage (from repo root):
    python scripts/regen_encoder_data.py --cell K562 --n-threads 12
"""
import argparse
import os
import sys
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from run_pipeline import load_config, load_sample_table          # noqa: E402
from preprocessing.abc.utils import StepLogger                    # noqa: E402
from preprocessing.abc.encoder_pretrain_data import generate_encoder_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--n-threads", type=int, default=12)
    args = ap.parse_args()

    repo_root = Path(_REPO)
    cfg = load_config(args.config)
    ref = cfg.get("reference", {})
    abc = cfg.get("abc_params", {})
    base_dir = Path(cfg.get("output", {}).get("base_dir", "./batch_output"))

    samples = load_sample_table(cfg["samples_table"], repo_root)
    try:
        sample = next(s for s in samples if s["cell_type"] == args.cell)
    except StopIteration:
        raise SystemExit(f"cell {args.cell} not found in samples table")

    peaks = sample.get("encoder_peaks_file")
    if not peaks:
        raise SystemExit(
            f"No encoder_peaks_file for {args.cell} in samples.tsv — add the ENCODE "
            "H3K27ac narrowPeak path (see reference/).")
    out_dir = str(base_dir / args.cell / "links")

    logger = StepLogger(1)
    logger.start_step(f"[{args.cell}] Regenerate encoder data from ENCODE H3K27ac peaks")
    logger.info(f"peaks   = {peaks}")
    logger.info(f"out_dir = {out_dir}")
    csv = generate_encoder_data(
        peaks, ref.get("fasta"), ref.get("chrom_sizes"),
        out_dir, logger,
        cell_type=args.cell,
        neg_fraction=abc.get("neg_fraction", 0.05),
        blacklist=ref.get("blacklist"),
        n_threads=args.n_threads,
        accessibility_bam=sample.get("accessibility_bam"),
        h3k27ac_bam=sample.get("h3k27ac_bam"),
        # replicate lists (mean-pooled RPM) take precedence when present; GM12878 uses 2/assay
        accessibility_bams=sample.get("accessibility_bams"),
        h3k27ac_bams=sample.get("h3k27ac_bams"),
        max_peaks=abc.get("max_encoder_peaks", 100_000),
    )
    logger.done()
    print("Wrote:", csv)


if __name__ == "__main__":
    main()
