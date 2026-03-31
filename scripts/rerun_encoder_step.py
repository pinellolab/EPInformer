#!/usr/bin/env python
"""Re-run ABC pipeline Step 4 only (generate_encoder_data) using config + samples.tsv.

Usage:
  conda run -n EPInformer_env python scripts/rerun_encoder_step.py --cell K562
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/rerun_encoder_step.py` from repo root
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from run_pipeline import load_config, load_sample_table
from preprocessing.abc.utils import StepLogger
from preprocessing.abc.encoder_pretrain_data import generate_encoder_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run encoder CSV generation (ABC step 4)")
    parser.add_argument("--config", default="config/config.yaml", help="YAML config path")
    parser.add_argument("--cell", required=True, help="cell_type from samples table (e.g. K562)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_dir = cfg["_project_dir"]
    samples = load_sample_table(cfg["samples_table"], project_dir)
    sample = next((s for s in samples if s["cell_type"] == args.cell), None)
    if sample is None:
        sys.exit(f"No sample with cell_type={args.cell!r}")

    ref = cfg["reference"]
    abc = cfg.get("abc_params", {})
    base_dir = Path(cfg["output"]["base_dir"])
    output_dir = str(base_dir / sample["cell_type"] / "links")

    peaks_file = sample.get("peaks_file")
    narrowpeak = peaks_file if peaks_file else str(Path(output_dir) / "macs2" / "peaks_peaks.narrowPeak")

    if not Path(narrowpeak).is_file():
        sys.exit(f"narrowPeak not found: {narrowpeak}")

    logger = StepLogger(1)
    logger.start_step("Generating sequence encoder training data (Step 4 only)")
    out = generate_encoder_data(
        narrowpeak,
        ref["fasta"],
        ref["chrom_sizes"],
        output_dir,
        logger,
        cell_type=sample["cell_type"],
        neg_fraction=abc.get("neg_fraction", 0.05),
        blacklist=ref.get("blacklist"),
        n_threads=abc.get("n_threads", 1),
        accessibility_bam=sample["accessibility_bam"],
        h3k27ac_bam=sample.get("h3k27ac_bam"),
        max_peaks=abc.get("max_encoder_peaks", 100_000),
    )
    logger.done()
    print("encoder_pretrain_data:", out, flush=True)


if __name__ == "__main__":
    main()
