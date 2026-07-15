#!/usr/bin/env python
"""
EPInformer batch pipeline -- config-driven orchestrator.

Reads a YAML config and TSV sample table, then runs:
  Stage 1 (links):    run_abc_pipeline()      BAM -> element-gene links
  Stage 2 (encoding): obtain_PE_withSignals()  links -> HDF5 for training
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required: pip install pyyaml"
    )

# ---------------------------------------------------------------------------
# Config & sample table loading
# ---------------------------------------------------------------------------

_PATH_KEYS_REFERENCE = {
    "fasta", "gene_bed", "chrom_sizes", "expression_csv", "blacklist",
    "average_hic_dir",
}
_PATH_KEYS_SAMPLE = {"accessibility_bam", "h3k27ac_bam", "hic_file", "qnorm_ref", "peaks_file", "encoder_peaks_file"}
# Optional multi-replicate BAM lists (comma/space-separated) used ONLY for the encoder
# activity, which mean-pools per-rep RPM across reps (reference recipe). Falls back to the
# single accessibility_bam/h3k27ac_bam when empty. GM12878 0.617 needs 2 reps/assay.
_PATH_LIST_KEYS_SAMPLE = {"accessibility_bams", "h3k27ac_bams"}
_REQUIRED_SAMPLE_COLS = {"cell_type", "accessibility_bam", "assay"}
_BOOL_SAMPLE_COLS = {"skip_links", "skip_encoding"}


def _resolve(path_str: Optional[str], base: Path) -> Optional[str]:
    """Resolve a path relative to *base* unless it is already absolute."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = base / p
    return str(p)


def load_config(config_path: str) -> dict:
    """Load YAML config and resolve relative paths.

    - ``samples_table`` is resolved relative to the config file directory.
    - All other paths (reference, output) are resolved relative to cwd
      (the project root where ``run_pipeline.py`` is invoked).
    """
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    project_dir = Path.cwd()

    with open(config_file) as fh:
        cfg = yaml.safe_load(fh)

    cfg["_config_dir"] = config_dir
    cfg["_project_dir"] = project_dir

    # Resolve reference paths against project root
    ref = cfg.get("reference", {})
    for key in _PATH_KEYS_REFERENCE:
        if key in ref and ref[key]:
            ref[key] = _resolve(ref[key], project_dir)

    # Resolve samples_table relative to config file
    if "samples_table" in cfg:
        cfg["samples_table"] = _resolve(cfg["samples_table"], config_dir)

    # Resolve output base_dir against project root
    out = cfg.get("output", {})
    if "base_dir" in out and out["base_dir"]:
        out["base_dir"] = _resolve(out["base_dir"], project_dir)

    return cfg


def load_sample_table(tsv_path: str, project_dir: Path) -> list[dict]:
    """Load TSV sample table. Resolve file-path columns against *project_dir*."""
    samples = []
    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row_num, row in enumerate(reader, start=2):
            # Strip whitespace from keys and values
            row = {k.strip(): (v.strip() if v else "") for k, v in row.items()}

            # Validate required columns
            missing = _REQUIRED_SAMPLE_COLS - set(row.keys())
            if missing:
                raise ValueError(f"Row {row_num}: missing required columns: {missing}")
            for col in _REQUIRED_SAMPLE_COLS:
                if not row[col]:
                    raise ValueError(f"Row {row_num}: required column '{col}' is empty")

            # Validate assay value
            if row["assay"] not in ("dnase", "atac"):
                raise ValueError(
                    f"Row {row_num}: assay must be 'dnase' or 'atac', got '{row['assay']}'"
                )

            # Resolve file paths
            for key in _PATH_KEYS_SAMPLE:
                if key in row and row[key]:
                    row[key] = _resolve(row[key], project_dir)
                elif key in row:
                    row[key] = None

            # Resolve multi-replicate BAM lists (comma/space-separated -> list of paths)
            for key in _PATH_LIST_KEYS_SAMPLE:
                val = row.get(key)
                if val and val.strip():
                    row[key] = [_resolve(p, project_dir)
                                for p in re.split(r"[,\s]+", val.strip()) if p]
                else:
                    row[key] = None

            # Parse boolean columns
            for key in _BOOL_SAMPLE_COLS:
                val = row.get(key, "").lower()
                row[key] = val in ("true", "1", "yes")

            samples.append(row)

    if not samples:
        raise ValueError(f"No samples found in {tsv_path}")

    return samples


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_links_stage(sample: dict, cfg: dict, dry_run: bool = False) -> dict:
    """Stage 1: element-gene links via run_abc_pipeline()."""
    ref = cfg.get("reference", {})
    abc = cfg.get("abc_params", {})
    base_dir = Path(cfg.get("output", {}).get("base_dir", "./batch_output"))
    output_dir = str(base_dir / sample["cell_type"] / "links")

    # Use the split ENCODE/rE2G average Hi-C reference when this cell type has
    # no usable cell-specific .hic file.  A real cell-specific file always wins.
    cell_hic = sample.get("hic_file")
    average_hic_dir = ref.get("average_hic_dir")
    if not cell_hic or not Path(cell_hic).exists():
        if average_hic_dir:
            reason = "not specified" if not cell_hic else f"not found ({cell_hic})"
            print(f"  [Hi-C] Cell-specific Hi-C {reason}; using average Hi-C: {average_hic_dir}")
            cell_hic = average_hic_dir

    kwargs = dict(
        accessibility_bam=sample["accessibility_bam"],
        output_dir=output_dir,
        assay=sample["assay"],
        h3k27ac_bam=sample.get("h3k27ac_bam"),
        # multi-rep BAM lists for the encoder activity (mean-pooled); None -> single-bam path
        accessibility_bams=sample.get("accessibility_bams"),
        h3k27ac_bams=sample.get("h3k27ac_bams"),
        hic_file=cell_hic,
        average_hic_dir=average_hic_dir,
        gene_bed=ref.get("gene_bed"),
        chrom_sizes=ref.get("chrom_sizes"),
        expression=ref.get("expression_csv"),
        expression_column=sample.get("expression_column") or None,
        fasta=ref.get("fasta"),
        qnorm_ref=sample.get("qnorm_ref"),
        cell_type=sample["cell_type"],
        preset=sample.get("preset") or None,
        peaks_file=sample.get("peaks_file"),
        encoder_peaks_file=sample.get("encoder_peaks_file"),
        blacklist=ref.get("blacklist"),
        # ABC algorithm params
        n_top_peaks=abc.get("n_top_peaks", 150_000),
        peak_extend=abc.get("peak_extend", 250),
        max_distance=abc.get("max_distance", 2_500_000),
        gamma=abc.get("gamma", 0.87),
        tss_slop=abc.get("tss_slop", 500),
        hic_resolution=abc.get("hic_resolution", 5000),
        neg_fraction=abc.get("neg_fraction", 0.05),
        max_encoder_peaks=abc.get("max_encoder_peaks", 100_000),
        include_self_promoter=abc.get("include_self_promoter", False),
        include_promoter_region=abc.get("include_promoter_region", False),
        n_threads=abc.get("n_threads", 1),
        # Hi-C processing params
        hic_gamma=abc.get("hic_gamma", 1.024238616787792),
        hic_scale=abc.get("hic_scale", 5.9594510043736655),
        hic_gamma_reference=abc.get("hic_gamma_reference", 0.87),
        hic_pseudocount_distance=abc.get("hic_pseudocount_distance", 5000),
        scale_hic_using_powerlaw=abc.get("scale_hic_using_powerlaw", True),
        tss_hic_contribution=abc.get("tss_hic_contribution", 100.0),
        dry_run=dry_run,
    )

    if dry_run:
        print(f"  [dry-run] Would call run_abc_pipeline() with:")
        for k, v in kwargs.items():
            if v is not None:
                print(f"    {k} = {v}")
        return {"output_dir": output_dir}

    from preprocessing.abc import run_abc_pipeline
    return run_abc_pipeline(**kwargs)


def run_encoding_stage(
    sample: dict, cfg: dict, links_outputs: dict, dry_run: bool = False
) -> Optional[str]:
    """Stage 2: HDF5 encoding via obtain_PE_withSignals()."""
    ref = cfg.get("reference", {})
    pre = cfg.get("preprocessing_params", {})
    base_dir = Path(cfg.get("output", {}).get("base_dir", "./batch_output"))
    links_dir = base_dir / sample["cell_type"] / "links"
    output_dir = str(base_dir / sample["cell_type"] / "encoding")

    # Locate link-stage outputs
    predictions = links_outputs.get(
        "predictions",
        str(links_dir / "Predictions" / "EnhancerPredictionsAllPutative.txt"),
    )
    _enh_default = links_dir / "EnhancerList.txt"
    if not _enh_default.exists():
        _enh_default = links_dir / "Neighborhoods" / "EnhancerList.txt"
    enhancer_list = links_outputs.get("enhancer_list", str(_enh_default))

    no_bigwig = pre.get("no_bigwig", True)
    signal_files = None if no_bigwig else pre.get("signal_files")

    kwargs = dict(
        fname=[predictions, enhancer_list],
        max_distance=pre.get("max_distance", 100_000),
        min_distance=pre.get("min_distance", 0),
        n_enhancer=pre.get("n_enhancer", 60),
        max_seq_len=pre.get("max_seq_len", 2000),
        gene_expression_csv=ref.get("expression_csv"),
        fasta_path=ref.get("fasta"),
        output_dir=output_dir,
        signal_files=signal_files,
        tss_column=pre.get("tss_column", "TSS_xpresso"),
        include_self_promoter=pre.get("include_self_promoter", True),
        cell_type=sample["cell_type"],
    )

    if dry_run:
        print(f"  [dry-run] Would call obtain_PE_withSignals() with:")
        for k, v in kwargs.items():
            if v is not None:
                print(f"    {k} = {v}")
        return None

    from preprocessing.pipelines_legacy import obtain_PE_withSignals
    obtain_PE_withSignals(**kwargs)
    return output_dir


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_sample(
    sample: dict, cfg: dict, stages: str, dry_run: bool = False
) -> dict:
    """Run requested stages for one sample. Returns status dict."""
    cell = sample["cell_type"]
    result = {"cell_type": cell, "status": "ok", "error": None}
    links_outputs: dict = {}

    # Stage 1: element-gene links
    if stages in ("links", "both") and not sample.get("skip_links"):
        print(f"\n{'='*60}")
        print(f"[{cell}] Stage 1: element-gene links")
        print(f"{'='*60}")
        links_outputs = run_links_stage(sample, cfg, dry_run=dry_run)
    else:
        if sample.get("skip_links"):
            print(f"\n[{cell}] Skipping links stage (skip_links=true)")

    # Stage 2: HDF5 encoding
    if stages in ("encoding", "both") and not sample.get("skip_encoding"):
        print(f"\n{'='*60}")
        print(f"[{cell}] Stage 2: HDF5 encoding")
        print(f"{'='*60}")
        run_encoding_stage(sample, cfg, links_outputs, dry_run=dry_run)
    else:
        if sample.get("skip_encoding"):
            print(f"\n[{cell}] Skipping encoding stage (skip_encoding=true)")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EPInformer batch pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python run_pipeline.py --config config/config.yaml
  python run_pipeline.py --config config/config.yaml --stages links --samples K562
  python run_pipeline.py --config config/config.yaml --dry-run
""",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--stages",
        choices=["links", "encoding", "both"],
        default="both",
        help="Which stages to run (default: both)",
    )
    parser.add_argument(
        "--samples",
        help="Comma-separated cell types to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print what would run",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first sample failure",
    )
    args = parser.parse_args()

    # Load config and samples
    cfg = load_config(args.config)
    project_dir = cfg["_project_dir"]
    samples = load_sample_table(cfg["samples_table"], project_dir)

    # Filter samples if requested
    if args.samples:
        keep = {s.strip() for s in args.samples.split(",")}
        samples = [s for s in samples if s["cell_type"] in keep]
        if not samples:
            sys.exit(f"No matching samples for: {args.samples}")

    # Print summary
    print(f"Config:  {args.config}")
    print(f"Samples: {len(samples)} [{', '.join(s['cell_type'] for s in samples)}]")
    print(f"Stages:  {args.stages}")
    if args.dry_run:
        print("Mode:    DRY RUN")
    print()

    # Process samples
    results = []
    t0 = time.time()

    for sample in samples:
        try:
            result = run_sample(sample, cfg, args.stages, dry_run=args.dry_run)
            results.append(result)
        except Exception as exc:
            results.append({
                "cell_type": sample["cell_type"],
                "status": "FAILED",
                "error": str(exc),
            })
            print(f"\n[{sample['cell_type']}] FAILED: {exc}")
            traceback.print_exc()
            if args.fail_fast:
                break

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    for r in results:
        status = r["status"]
        cell = r["cell_type"]
        if status == "ok":
            print(f"  {cell}: OK")
        else:
            print(f"  {cell}: FAILED -- {r['error']}")

    n_failed = sum(1 for r in results if r["status"] != "ok")
    if n_failed:
        sys.exit(f"\n{n_failed} sample(s) failed.")


if __name__ == "__main__":
    main()
