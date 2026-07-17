"""Canonical EPInformer-seq-v2 model and lightweight inference helpers.

EPInformer-seq-v2 consumes 2,114 bp and predicts DNase and H3K27ac profiles
over the central 1,024 bp. Each cell type uses its own main model and frozen
sequence-bias model. Downstream applications such as ``pinellolab/chorus``
should reference this implementation as the upstream model definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROFILE_WINDOW = 1024
INPUT_WINDOW = 2114
AVAILABLE_CELL_TYPES = (
    "K562", "GM12878", "HepG2", "A549", "HeLa", "HMEC", "HSMM",
    "HUVEC", "NHEK", "NHLF", "H1",
)
AVAILABLE_ASSAYS = (
    "Enhancer_DNase",
    "Enhancer_H3K27ac",
    "Enhancer_H3K27ac_DNase",
)

__all__ = [
    "PROFILE_WINDOW",
    "INPUT_WINDOW",
    "AVAILABLE_CELL_TYPES",
    "AVAILABLE_ASSAYS",
    "DilatedResBlock",
    "EPInformerSeqV2",
    "EPInformerSeqV2Bias",
    "EPInformerSeq",
    "EPInformerSeqBias",
    "one_hot_dna",
    "reverse_complement_one_hot",
    "load_models",
    "predict_profile",
    "activity_from_profiles",
    "predict_activity",
    "multinomial_nll",
    "count_mse",
]

_ALPHABET = {ord(base): index for index, base in enumerate("ACGT")}


class DilatedResBlock(nn.Module):
    """Length-preserving dilated convolution with a residual connection."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.bn(self.conv(x)) + x)


class EPInformerSeqV2(nn.Module):
    """EPInformer-seq-v2 per-cell 2,114-bp sequence-to-profile model."""

    def __init__(
        self,
        stem_channels: int | None = None,
        body_channels: int | None = None,
        n_dilated: int = 9,
        input_window: int = INPUT_WINDOW,
        output_window: int = PROFILE_WINDOW,
        *,
        stem_ch: int | None = None,
        body_ch: int | None = None,
        in_window: int | None = None,
        out_window: int | None = None,
    ):
        super().__init__()
        stem_channels = 128 if stem_channels is None else stem_channels
        body_channels = 64 if body_channels is None else body_channels
        if stem_ch is not None: stem_channels = stem_ch
        if body_ch is not None: body_channels = body_ch
        if in_window is not None: input_window = in_window
        if out_window is not None: output_window = out_window
        if input_window < output_window:
            raise ValueError("input_window must be at least output_window")
        self.input_window = input_window
        self.output_window = output_window
        self.stem = nn.Sequential(
            nn.Conv1d(4, stem_channels, kernel_size=21, padding=10),
            nn.ELU(),
            nn.Conv1d(stem_channels, body_channels, kernel_size=1),
            nn.ELU(),
        )
        self.blocks = nn.ModuleList(
            DilatedResBlock(body_channels, 2**index)
            for index in range(n_dilated)
        )
        self.profile_head = nn.Conv1d(body_channels, 2, kernel_size=1)
        self.count_head = nn.Sequential(
            nn.Linear(body_channels, 64), nn.SiLU(), nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        start = (h.shape[-1] - self.output_window) // 2
        h = h[:, :, start : start + self.output_window]
        profile_logits = self.profile_head(h)
        pooled = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        return profile_logits, self.count_head(pooled)


class EPInformerSeqV2Bias(nn.Module):
    """Frozen EPInformer-seq-v2 sequence-bias model used at inference."""

    def __init__(self, stem_channels: int | None = None, body_channels: int | None = None,
                 n_dilated: int = 9, *, stem_ch: int | None = None,
                 body_ch: int | None = None):
        super().__init__()
        stem_channels = 64 if stem_channels is None else stem_channels
        body_channels = 32 if body_channels is None else body_channels
        if stem_ch is not None: stem_channels = stem_ch
        if body_ch is not None: body_channels = body_ch
        self.stem = nn.Sequential(
            nn.Conv1d(4, stem_channels, kernel_size=21, padding=10),
            nn.ELU(),
            nn.Conv1d(stem_channels, body_channels, kernel_size=1),
            nn.ELU(),
        )
        self.blocks = nn.ModuleList(
            DilatedResBlock(body_channels, 2**index)
            for index in range(n_dilated)
        )
        self.profile_head = nn.Conv1d(body_channels, 2, kernel_size=1)
        self.count_head = nn.Sequential(
            nn.Linear(body_channels, 32), nn.SiLU(), nn.Linear(32, 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        profile_logits = self.profile_head(h)
        pooled = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        return profile_logits, self.count_head(pooled)


def multinomial_nll(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Normalized multinomial profile loss used to train EPInformer-seq-v2."""
    log_probabilities = F.log_softmax(logits, dim=-1)
    negative_log_likelihood = -(counts.float() * log_probabilities).sum(dim=-1)
    totals = counts.float().sum(dim=-1).clamp(min=1.0)
    return (negative_log_likelihood / totals).mean()


def count_mse(log_prediction: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and observed log10 total counts."""
    return F.mse_loss(log_prediction, torch.log10(observed.float() + 1.0))


def one_hot_dna(sequence: str, length: int = INPUT_WINDOW) -> np.ndarray:
    """Encode A/C/G/T as ``(4, length)`` float32; unknown bases are zero."""
    sequence = sequence.strip().upper()
    if len(sequence) > length:
        excess = len(sequence) - length
        sequence = sequence[excess // 2 : excess // 2 + length]
    encoded = np.zeros((4, length), dtype=np.float32)
    for position, base in enumerate(sequence):
        channel = _ALPHABET.get(ord(base))
        if channel is not None:
            encoded[channel, position] = 1.0
    return encoded


def reverse_complement_one_hot(encoded: np.ndarray) -> np.ndarray:
    return encoded[::-1, ::-1].copy()


def _checkpoint_state(path: str | Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return state


def load_models(
    main_weights: str | Path,
    bias_weights: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[EPInformerSeqV2, EPInformerSeqV2Bias]:
    """Load a matching per-cell ``main.pt`` and ``bias.pt`` checkpoint pair."""
    main = EPInformerSeqV2()
    main.load_state_dict(_checkpoint_state(main_weights))
    bias = EPInformerSeqV2Bias()
    bias.load_state_dict(_checkpoint_state(bias_weights))
    for parameter in bias.parameters():
        parameter.requires_grad_(False)
    return main.eval().to(device), bias.eval().to(device)


def predict_profile(
    main: nn.Module,
    bias: nn.Module,
    sequence: str,
    *,
    device: str | torch.device = "cpu",
    average_reverse_complement: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return DNase profile, H3K27ac profile, and two predicted counts."""
    encoded = one_hot_dna(sequence, INPUT_WINDOW)
    crop_start = (INPUT_WINDOW - PROFILE_WINDOW) // 2

    def run(item: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        main_input = torch.from_numpy(item).unsqueeze(0).to(device)
        bias_input = torch.from_numpy(
            item[:, crop_start : crop_start + PROFILE_WINDOW].copy()
        ).unsqueeze(0).to(device)
        main_profile, log_counts = main(main_input)
        bias_profile, _ = bias(bias_input)
        counts = 10.0**log_counts
        signal = torch.softmax(main_profile + bias_profile, dim=-1)
        return signal * counts.unsqueeze(-1), counts

    with torch.inference_mode():
        signal, counts = run(encoded)
        if average_reverse_complement:
            reverse_signal, reverse_counts = run(reverse_complement_one_hot(encoded))
            signal = 0.5 * (signal + torch.flip(reverse_signal, dims=[-1]))
            counts = 0.5 * (counts + reverse_counts)
    signal_np = signal[0].cpu().numpy().astype(np.float32)
    return signal_np[0], signal_np[1], counts[0].cpu().numpy().astype(np.float32)


def predict_activity(
    main: nn.Module,
    bias: nn.Module,
    sequence: str,
    assay: str = "Enhancer_DNase",
    *,
    device: str | torch.device = "cpu",
    average_reverse_complement: bool = True,
) -> float:
    """Aggregate the central 256 bp into one enhancer-activity score."""
    dnase, h3k27ac, _ = predict_profile(
        main, bias, sequence, device=device,
        average_reverse_complement=average_reverse_complement,
    )
    return activity_from_profiles(dnase, h3k27ac, assay)


def activity_from_profiles(
    dnase: np.ndarray,
    h3k27ac: np.ndarray,
    assay: str = "Enhancer_DNase",
) -> float:
    """Aggregate two 1,024-bp profiles over their central 256 bp."""
    if assay not in AVAILABLE_ASSAYS:
        raise ValueError(f"Unknown assay {assay!r}; choose from {AVAILABLE_ASSAYS}")
    if dnase.shape[-1] != PROFILE_WINDOW or h3k27ac.shape[-1] != PROFILE_WINDOW:
        raise ValueError(f"profiles must each contain {PROFILE_WINDOW} positions")
    start = (PROFILE_WINDOW - 256) // 2
    dnase_max = float(np.max(dnase[start : start + 256]))
    h3k27ac_max = float(np.max(h3k27ac[start : start + 256]))
    if assay == "Enhancer_DNase":
        return dnase_max
    if assay == "Enhancer_H3K27ac":
        return h3k27ac_max
    return float(np.sqrt(dnase_max * h3k27ac_max + 1e-12))


# Backward-compatible aliases for code written before the v2 name was made
# explicit in this repository.
EPInformerSeq = EPInformerSeqV2
EPInformerSeqBias = EPInformerSeqV2Bias
