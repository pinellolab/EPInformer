#!/usr/bin/env python3
"""
General preprocessing CLI for any cell type.

Wraps :func:`epinformer_preprocessing.pipelines_legacy.obtain_PE_withSignals`
to generate ``samples.h5`` (factored HDF5) from ABC pipeline outputs.

Examples::

    # K562 with no BigWig (sequence + ABC features only)
    python run_preprocessing.py with-signals --no-bigwig \\
        --cell-type K562 --output-dir ./training_data/K562_run \\
        --predictions ./abc_output/K562/Predictions/EnhancerPredictionsAllPutative.txt \\
        --enhancer-list ./abc_output/K562/EnhancerList.txt \\
        --include-self-promoter

    # GM12878 with BigWig signals
    python run_preprocessing.py with-signals \\
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
    return Path(__file__).resolve().parent / "data_EPInformer"


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


def _add_shared(p: argparse.ArgumentParser, default_output: str) -> None:
    ed = _default_epinformer_data()
    p.add_argument(
        "--gene-expr-csv",
        default=str(ed / "GM12878_K562_18377_gene_expr_fromXpresso.csv"),
        help="Gene expression / features table (must contain Actual_{cell_type} column).",
    )
    p.add_argument(
        "--tss-column",
        default="TSS_xpresso",
        help="Column to treat as TSS (renamed to TSS internally).",
    )
    p.add_argument(
        "--fasta",
        default=str(ed / "hg38.fa"),
        help="Reference genome FASTA (hg38).",
    )
    p.add_argument(
        "--output-dir",
        default=default_output,
        help="Output directory (will contain samples.h5 and CSV sidecars).",
    )
    p.add_argument(
        "--cell-type",
        default="K562",
        choices=_CELL_TYPES,
        help="Cell type — selects expression column Actual_{cell_type}.",
    )


def cmd_with_signals(args: argparse.Namespace) -> None:
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

    from epinformer_preprocessing.pipelines_legacy import obtain_PE_withSignals

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate EPInformer training HDF5 from ABC pipeline outputs (any cell type)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- with-signals subcommand ---
    p_ws = sub.add_parser(
        "with-signals",
        help="Multi-track BigWig pipeline (obtain_PE_withSignals)",
    )
    _add_shared(p_ws, default_output=str(REPO_ROOT / "training_data" / "run"))
    p_ws.add_argument(
        "--predictions", required=True,
        help="EnhancerPredictionsAllPutative (.txt/.txt.gz) from ABC.",
    )
    p_ws.add_argument(
        "--enhancer-list", required=True,
        help="EnhancerList.txt from ABC Neighborhoods.",
    )
    sig_ws = p_ws.add_mutually_exclusive_group(required=True)
    sig_ws.add_argument(
        "--signal-bigwigs",
        nargs="+",
        metavar="PATH",
        help="One or more BigWig files (order = channel order in seq_signal).",
    )
    sig_ws.add_argument(
        "--no-bigwig",
        action="store_true",
        help="Sequence + ABC tabular features only; omit seq_signal from samples.h5.",
    )
    p_ws.add_argument("--min-distance", type=int, default=0, help="Minimum abs(distance) to TSS in bp.")
    p_ws.add_argument("--max-distance", type=int, default=100_000, help="Maximum abs(distance) to TSS in bp.")
    p_ws.add_argument("--n-enhancer", type=int, default=60, help="Max enhancer elements per gene.")
    p_ws.add_argument("--max-seq-len", type=int, default=2000)
    p_ws.add_argument("--add-flank", action="store_true")
    p_ws.add_argument("--include-self-promoter", action="store_true",
                       help="Include isSelfPromoter elements from ABC all-putative file.")
    p_ws.add_argument("--abc-all-putative", default=None,
                       help="Path to EnhancerPredictionsAllPutative (for self-promoter data).")
    p_ws.set_defaults(func=cmd_with_signals)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
