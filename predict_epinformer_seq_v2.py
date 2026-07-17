#!/usr/bin/env python3
"""Run EPInformer-seq-v2 on one 2,114-bp DNA sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from EPInformer.epinformer_seq_v2 import (
    AVAILABLE_ASSAYS,
    INPUT_WINDOW,
    activity_from_profiles,
    load_models,
    predict_profile,
)


def read_sequence(value: str | None, fasta: Path | None) -> str:
    if (value is None) == (fasta is None):
        raise ValueError("provide exactly one of --sequence or --fasta")
    if value is not None:
        return value.strip().upper()
    lines = [line.strip() for line in fasta.read_text().splitlines() if not line.startswith(">")]
    return "".join(lines).upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence", help="DNA sequence (center-cropped or padded to 2,114 bp)")
    source.add_argument("--fasta", type=Path, help="single-record FASTA file")
    parser.add_argument("--main-weights", type=Path, required=True)
    parser.add_argument("--bias-weights", type=Path, required=True)
    parser.add_argument("--assay", choices=AVAILABLE_ASSAYS, default="Enhancer_DNase")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-reverse-complement", action="store_true")
    parser.add_argument("--output", type=Path, help="optional .npz file for both profiles and counts")
    args = parser.parse_args()

    sequence = read_sequence(args.sequence, args.fasta)
    if len(sequence) != INPUT_WINDOW:
        print(f"Note: input length {len(sequence):,}; encoding will center-crop/pad to {INPUT_WINDOW:,} bp")
    main_model, bias_model = load_models(args.main_weights, args.bias_weights, args.device)
    average_rc = not args.no_reverse_complement
    dnase, h3k27ac, counts = predict_profile(
        main_model, bias_model, sequence, device=args.device,
        average_reverse_complement=average_rc,
    )
    activity = activity_from_profiles(dnase, h3k27ac, args.assay)
    print(f"{args.assay}\t{activity:.8g}")
    print(f"predicted_counts\tDNase={counts[0]:.8g}\tH3K27ac={counts[1]:.8g}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output, dnase=dnase, h3k27ac=h3k27ac, counts=counts,
            assay=np.array(args.assay), activity=np.float32(activity),
        )


if __name__ == "__main__":
    main()
