#!/usr/bin/env python3
"""Run the original 256-bp EPInformer-seq enhancer-activity model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from EPInformer.models import enhancer_predictor_256bp


INPUT_WINDOW = 256
_BASE_TO_CHANNEL = {base: index for index, base in enumerate("ACGT")}


def read_sequence(sequence: str | None, fasta: Path | None) -> str:
    """Read a literal sequence or one single-record FASTA."""
    if sequence is not None:
        return sequence.strip().upper()
    lines = [
        line.strip()
        for line in fasta.read_text().splitlines()
        if line.strip() and not line.startswith(">")
    ]
    return "".join(lines).upper()


def one_hot_dna(sequence: str, length: int = INPUT_WINDOW) -> np.ndarray:
    """Center-crop/pad DNA to an A/C/G/T one-hot array of shape ``(4, L)``."""
    if len(sequence) > length:
        excess = len(sequence) - length
        sequence = sequence[excess // 2 : excess // 2 + length]
    encoded = np.zeros((4, length), dtype=np.float32)
    for position, base in enumerate(sequence):
        channel = _BASE_TO_CHANNEL.get(base)
        if channel is not None:
            encoded[channel, position] = 1.0
    return encoded


def load_model(weights: Path, device: str) -> enhancer_predictor_256bp:
    """Load an EPInformer-seq training checkpoint or raw state dictionary."""
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = enhancer_predictor_256bp()
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def predict(model, sequence: str, device: str, average_reverse_complement: bool = True):
    """Return ``(log2(0.1 + activity), activity)`` for one sequence."""
    encoded = one_hot_dna(sequence)

    def run(item: np.ndarray) -> float:
        tensor = torch.from_numpy(item).unsqueeze(0).to(device)
        return float(model(tensor).reshape(-1)[0].cpu())

    with torch.inference_mode():
        prediction_log2 = run(encoded)
        if average_reverse_complement:
            prediction_log2 = 0.5 * (prediction_log2 + run(encoded[::-1, ::-1].copy()))
    activity = max(2.0**prediction_log2 - 0.1, 0.0)
    return prediction_log2, activity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence", help="DNA sequence; center-cropped/padded to 256 bp")
    source.add_argument("--fasta", type=Path, help="single-record FASTA file")
    parser.add_argument("--weights", type=Path, required=True, help="original EPInformer-seq checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-reverse-complement", action="store_true")
    args = parser.parse_args()

    sequence = read_sequence(args.sequence, args.fasta)
    if len(sequence) != INPUT_WINDOW:
        print(f"Note: input length {len(sequence):,}; center-cropping/padding to {INPUT_WINDOW} bp")
    model = load_model(args.weights, args.device)
    prediction_log2, activity = predict(
        model,
        sequence,
        args.device,
        average_reverse_complement=not args.no_reverse_complement,
    )
    print(f"log2_0.1_plus_activity\t{prediction_log2:.8g}")
    print(f"H3K27ac_DNase_activity\t{activity:.8g}")


if __name__ == "__main__":
    main()
