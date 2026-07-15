#!/usr/bin/env python3
"""
General preprocessing CLI for any cell type.

Wraps :func:`preprocessing.pipelines_legacy.obtain_PE_withSignals`
to generate ``samples.h5`` (factored HDF5) from ABC pipeline outputs.

Examples::

    # K562 with no BigWig (sequence + ABC features only)
    python run_preprocessing.py --no-bigwig \\
        --cell-type K562 --output-dir ./training_data/K562_run \\
        --predictions ./abc_output/K562/Predictions/EnhancerPredictionsAllPutative.txt \\
        --enhancer-list ./abc_output/K562/EnhancerList.txt \\
        --include-self-promoter

    # GM12878 with BigWig signals
    python run_preprocessing.py \\
        --cell-type GM12878 --output-dir ./training_data/GM12878_run \\
        --predictions ./abc_output/GM12878/Predictions/EnhancerPredictionsAllPutative.txt \\
        --enhancer-list ./abc_output/GM12878/EnhancerList.txt \\
        --signal-bigwigs dnase.bigWig h3k27ac.bigWig
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_CELL_TYPES = ["K562", "GM12878", "H1", "HUVEC", "NHEK", "HepG2"]


def _default_epinformer_data() -> Path:
    return Path(__file__).resolve().parent / "data"


def _resolve(p: str | None, cwd: Path) -> str | None:
    if p is None:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = (cwd / path).resolve()
    return str(path)


def _must_exist(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise SystemExit(f"Missing {label}: {path}")


def _ensure_bigwigs(paths: list[str], label: str = "BigWig") -> None:
    for p in paths:
        _must_exist(p, label)


def main() -> None:
    ed = _default_epinformer_data()

    parser = argparse.ArgumentParser(
        description="Generate EPInformer training HDF5 from ABC pipeline outputs (any cell type)"
    )
    parser.add_argument(
        "--gene-expr-csv",
        default=str(ed / "GM12878_K562_18377_gene_expr_fromXpresso.csv"),
        help="Gene expression / features table (must contain {cell_type}_RPKM column).",
    )
    parser.add_argument(
        "--tss-column",
        default="TSS_xpresso",
        help="Column to treat as TSS (renamed to TSS internally).",
    )
    parser.add_argument(
        "--fasta",
        default=str(ed / "reference" / "hg38" / "hg38.fa"),
        help="Reference genome FASTA (hg38).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "training_data" / "run"),
        help="Output directory (will contain samples.h5 and CSV sidecars).",
    )
    parser.add_argument(
        "--cell-type",
        default="K562",
        choices=_CELL_TYPES,
        help="Cell type — selects expression column {cell_type}_RPKM.",
    )
    parser.add_argument(
        "--predictions", required=True,
        help="EnhancerPredictionsAllPutative (.txt/.txt.gz) from ABC.",
    )
    parser.add_argument(
        "--enhancer-list", required=True,
        help="EnhancerList.txt from ABC Neighborhoods.",
    )
    sig = parser.add_mutually_exclusive_group(required=True)
    sig.add_argument(
        "--signal-bigwigs",
        nargs="+",
        metavar="PATH",
        help="One or more BigWig files (order = channel order in seq_signal).",
    )
    sig.add_argument(
        "--no-bigwig",
        action="store_true",
        help="Sequence + ABC tabular features only; omit seq_signal from samples.h5.",
    )
    parser.add_argument("--min-distance", type=int, default=0, help="Minimum abs(distance) to TSS in bp.")
    parser.add_argument("--max-distance", type=int, default=100_000, help="Maximum abs(distance) to TSS in bp.")
    parser.add_argument("--n-enhancer", type=int, default=60, help="Max enhancer elements per gene.")
    parser.add_argument("--max-seq-len", type=int, default=2000)
    parser.add_argument("--add-flank", action="store_true")
    parser.add_argument("--include-self-promoter", action="store_true",
                        help="Include isSelfPromoter elements from ABC all-putative file.")
    parser.add_argument("--abc-all-putative", default=None,
                        help="Path to EnhancerPredictionsAllPutative (for self-promoter data).")

    args = parser.parse_args()

    cwd = Path.cwd()
    out = _resolve(args.output_dir, cwd)
    assert out is not None
    ge = _resolve(args.gene_expr_csv, cwd)
    fa = _resolve(args.fasta, cwd)
    pred = _resolve(args.predictions, cwd)
    enh = _resolve(args.enhancer_list, cwd)
    assert ge and fa and pred and enh
    _must_exist(ge, "gene expression CSV")
    _must_exist(fa, "FASTA")
    _must_exist(pred, "ABC predictions")
    _must_exist(enh, "EnhancerList")
    if args.no_bigwig:
        signal_bigwigs: list[str] = []
    else:
        signal_bigwigs = list(args.signal_bigwigs)
        _ensure_bigwigs(signal_bigwigs)

    from preprocessing.pipelines_legacy import obtain_PE_withSignals

    os.makedirs(out, exist_ok=True)
    print(f"Cell type:      {args.cell_type}")
    print(f"Gene expr:      {ge}")
    print(f"FASTA:          {fa}")
    print(f"Predictions:    {pred}")
    print(f"EnhancerList:   {enh}")
    print(f"Output:         {out}")
    print(f"Signal BigWigs: {signal_bigwigs if signal_bigwigs else '(none)'}")
    print(f"samples.h5 ->   {os.path.join(out, 'samples.h5')}")

    abc_putative = _resolve(args.abc_all_putative, cwd) if args.abc_all_putative else None

    obtain_PE_withSignals(
        [pred, enh],
        max_distance=args.max_distance,
        min_distance=args.min_distance,
        add_flank=args.add_flank,
        n_enhancer=args.n_enhancer,
        max_seq_len=args.max_seq_len,
        gene_expression_csv=ge,
        fasta_path=fa,
        output_dir=out,
        signal_files=signal_bigwigs,
        tss_column=args.tss_column,
        include_self_promoter=args.include_self_promoter,
        abc_all_putative=abc_putative,
        cell_type=args.cell_type,
    )


if __name__ == "__main__":
    main()
