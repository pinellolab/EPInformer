#!/usr/bin/env python3
"""
K562 preprocessing CLI (wraps :mod:`epinformer_preprocessing.pipelines_legacy`).

Run from the repository root (or anywhere, using absolute paths).

Examples::

    # Multi-track signals (5 BigWigs typical: DNase, H3K27ac, H3K4me1, H3K4me3, CTCF)
    python scripts/run_k562_preprocessing.py with-signals \\
        --output-dir ./training_data/k562_multitrack \\
        --signal-bigwigs /path/to/ENCFF972GVB.bigWig \\
        --signal-bigwigs /path/to/ENCFF465GBD.bigWig \\
        --signal-bigwigs /path/to/ENCFF287LBI.bigWig \\
        --signal-bigwigs /path/to/ENCFF144MRB.bigWig \\
        --signal-bigwigs /path/to/ENCFF168IFW.bigWig

    # Single DNase BigWig (H3K27ac overlap ABC pipeline)
    python scripts/run_k562_preprocessing.py h3k27ac \\
        --output-dir ./training_data/k562_dnase \\
        --dnase-bigwig /path/to/ENCFF972GVB.bigWig \\
        --predictions epinformer_data_20250503/.../EnhancerPredictionsAllPutative.avghic.txt

    # General obtain_PE (same gene merge as with-signals, multi-track)
    python scripts/run_k562_preprocessing.py pe \\
        --output-dir ./training_data/k562_pe \\
        --signal-bigwigs /path/to/dnase.bigWig --signal-bigwigs /path/to/h3k27ac.bigWig

Encode accessions often used in legacy paths: ENCFF972GVB (DNase), ENCFF465GBD (H3K27ac),
ENCFF287LBI (H3K4me1), ENCFF144MRB (H3K4me3), ENCFF168IFW (CTCF).

Omit BigWigs (FASTA + ABC features only; no ``seq_signal`` in HDF5)::

    python scripts/run_k562_preprocessing.py with-signals --no-bigwig --output-dir ./out_seq_only
    python scripts/run_k562_preprocessing.py h3k27ac --no-bigwig --output-dir ./out_seq_only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _add_shared(p: argparse.ArgumentParser) -> None:
    ed = _default_epinformer_data()
    p.add_argument(
        "--gene-expr-csv",
        default=str(ed / "GM12878_K562_18377_gene_expr_fromXpresso.csv"),
        help="Gene expression / features table (merged GM12878+K562 style).",
    )
    p.add_argument(
        "--tss-column",
        default="TSS_xpresso",
        help="Column to treat as TSS (renamed to TSS internally). Use TSS for pinello-style CSVs.",
    )
    p.add_argument(
        "--fasta",
        default=str(ed / "hg38.fa"),
        help="Reference genome FASTA (hg38).",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "training_data" / "k562_run"),
        help="Output directory (will contain samples.h5 and CSV sidecars).",
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

    from epinformer_preprocessing.pipelines_legacy import obtain_K562_PE_withSignals

    os.makedirs(out, exist_ok=True)
    print("gene_expr:", ge)
    print("fasta:", fa)
    print("predictions:", pred)
    print("enhancer_list:", enh)
    print("output_dir:", out)
    print("signal BigWigs:", signal_bigwigs if signal_bigwigs else "(none)")
    print("samples.h5 ->", os.path.join(out, "samples.h5"))

    abc_putative = _resolve(args.abc_all_putative, cwd) if args.abc_all_putative else None

    obtain_K562_PE_withSignals(
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
    )


def cmd_h3k27ac(args: argparse.Namespace) -> None:
    cwd = Path.cwd()
    out = _resolve(args.output_dir, cwd)
    assert out is not None
    ge = _resolve(args.gene_expr_csv, cwd)
    fa = _resolve(args.fasta, cwd)
    pred = _resolve(args.predictions, cwd)
    assert ge and fa and pred
    _must_exist(ge, "gene expression CSV")
    _must_exist(fa, "FASTA")
    _must_exist(pred, "ABC predictions")
    if args.no_bigwig:
        bw_resolved = None
    else:
        bw_resolved = _resolve(args.dnase_bigwig, cwd)
        assert bw_resolved
        _must_exist(bw_resolved, "DNase BigWig")

    from epinformer_preprocessing.pipelines_legacy import obtain_K562_H3K27ac_PE

    os.makedirs(out, exist_ok=True)
    print("gene_expr:", ge)
    print("fasta:", fa)
    print("predictions:", pred)
    print("dnase_bigwig:", bw_resolved if bw_resolved else "(none)")
    print("output_dir:", out)
    print("samples.h5 ->", os.path.join(out, "samples.h5"))

    obtain_K562_H3K27ac_PE(
        predictions_tsv=pred,
        gene_expression_csv=ge,
        fasta_path=fa,
        dnase_bigwig=bw_resolved,
        output_dir=out,
        tss_column=args.tss_column,
        include_bigwig=not args.no_bigwig,
    )


def cmd_pe(args: argparse.Namespace) -> None:
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
        signal_bigwigs = []
    else:
        signal_bigwigs = list(args.signal_bigwigs)
        _ensure_bigwigs(signal_bigwigs)

    from epinformer_preprocessing.pipelines_legacy import obtain_PE

    os.makedirs(out, exist_ok=True)
    print("gene_expr:", ge)
    print("fasta:", fa)
    print("predictions:", pred)
    print("enhancer_list:", enh)
    print("output_dir:", out)
    print("signal BigWigs:", signal_bigwigs if signal_bigwigs else "(none)")
    print("samples.h5 ->", os.path.join(out, "samples.h5"))

    obtain_PE(
        [pred, enh],
        signal_bigwigs,
        max_distance=args.max_distance,
        add_flank=args.add_flank,
        use_strand=args.use_strand,
        n_enhancer=args.n_enhancer,
        max_seq_len=args.max_seq_len,
        pe_type=args.pe_type,
        cell_type=args.cell_type,
        gene_expression_csv=ge,
        fasta_path=fa,
        output_dir=out,
        tss_column=args.tss_column,
    )


def main() -> None:
    ed = _default_epinformer_data()
    default_pred = (
        ed
        / "K562_ABC_EGLinks"
        / "gene_enhancer_links"
        / "EnhancerPredictionsAllPutative.avghic.txt"
    )
    default_enh = (
        ed
        / "K562_ABC_EGLinks"
        / "DNase_ENCFF257HEE_Neighborhoods"
        / "EnhancerList.txt"
    )

    parser = argparse.ArgumentParser(
        description="Run K562 promoter–enhancer preprocessing to compressed samples.h5"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ws = sub.add_parser(
        "with-signals",
        help="K562 multi-track BigWig pipeline (obtain_K562_PE_withSignals)",
    )
    _add_shared(p_ws)
    p_ws.add_argument(
        "--predictions",
        default=str(default_pred),
        help="EnhancerPredictionsAllPutative (.txt/.txt.gz) from ABC.",
    )
    p_ws.add_argument(
        "--enhancer-list",
        default=str(default_enh),
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
        help="Sequence + ABC tabular features only; omit seq_signal from samples.h5 (no BigWig reads).",
    )
    p_ws.add_argument("--min-distance", type=int, default=0, help="Minimum abs(distance) to TSS in bp.")
    p_ws.add_argument("--max-distance", type=int, default=100_000, help="Maximum abs(distance) to TSS in bp.")
    p_ws.add_argument("--n-enhancer", type=int, default=60, help="Max enhancer elements per gene.")
    p_ws.add_argument("--max-seq-len", type=int, default=2000)
    p_ws.add_argument("--add-flank", action="store_true")
    p_ws.add_argument("--include-self-promoter", action="store_true",
                       help="Include isSelfPromoter elements from ABC all-putative file.")
    p_ws.add_argument("--abc-all-putative", default=None,
                       help="Path to EnhancerPredictionsAllPutative.avghic.txt (for self-promoter data).")
    p_ws.set_defaults(func=cmd_with_signals)

    p_h3 = sub.add_parser(
        "h3k27ac",
        help="K562 single DNase BigWig pipeline (obtain_K562_H3K27ac_PE)",
    )
    _add_shared(p_h3)
    p_h3.add_argument(
        "--predictions",
        default=str(default_pred),
        help="EnhancerPredictionsAllPutative file for this run.",
    )
    sig_h3 = p_h3.add_mutually_exclusive_group(required=True)
    sig_h3.add_argument(
        "--dnase-bigwig",
        metavar="PATH",
        help="Single DNase-seq BigWig (e.g. ENCFF972GVB.bigWig).",
    )
    sig_h3.add_argument(
        "--no-bigwig",
        action="store_true",
        help="Encode sequences only; no DNase track in samples.h5.",
    )
    p_h3.set_defaults(func=cmd_h3k27ac)

    p_pe = sub.add_parser(
        "pe",
        help="General obtain_PE with multi-track BigWigs (relative distance, normalized_h3K27ac/dhs cols)",
    )
    _add_shared(p_pe)
    p_pe.add_argument(
        "--predictions",
        default=str(default_pred),
        help="EnhancerPredictionsAllPutative from ABC.",
    )
    p_pe.add_argument(
        "--enhancer-list",
        default=str(default_enh),
        help="EnhancerList.txt from ABC Neighborhoods.",
    )
    sig_pe = p_pe.add_mutually_exclusive_group(required=True)
    sig_pe.add_argument(
        "--signal-bigwigs",
        nargs="+",
        metavar="PATH",
        help="One or more BigWig files.",
    )
    sig_pe.add_argument(
        "--no-bigwig",
        action="store_true",
        help="obtain_PE without BigWig tracks (no seq_signal dataset).",
    )
    p_pe.add_argument("--max-distance", type=int, default=150_000)
    p_pe.add_argument("--n-enhancer", type=int, default=60)
    p_pe.add_argument("--max-seq-len", type=int, default=2000)
    p_pe.add_argument("--add-flank", action="store_true")
    p_pe.add_argument("--use-strand", action="store_true")
    p_pe.add_argument("--pe-type", default="AllPutative")
    p_pe.add_argument("--cell-type", default="K562")
    p_pe.set_defaults(func=cmd_pe)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
