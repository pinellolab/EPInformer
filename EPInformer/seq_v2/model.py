"""Training-facing names for the canonical EPInformer-seq-v2 architecture.

The module aliases the inference implementation so training and inference
cannot silently drift apart. State-dict keys remain compatible with Chorus's
``PerCellProfileNetWide`` and ``BiasNet`` checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..epinformer_seq_v2 import (
    DilatedResBlock,
    EPInformerSeqV2 as PerCellProfileNetWide,
    EPInformerSeqV2Bias as BiasNet,
)


def multinomial_nll(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Per-example multinomial negative log likelihood."""
    logp = F.log_softmax(logits, dim=-1)
    nll = -(counts.float() * logp).sum(dim=-1)
    total = counts.float().sum(dim=-1).clamp(min=1.0)
    return (nll / total).mean()


def count_mse(log_pred: torch.Tensor, total_obs: torch.Tensor) -> torch.Tensor:
    """MSE in the model's log10(total + 1) count space."""
    return F.mse_loss(log_pred, torch.log10(total_obs.float() + 1.0))


__all__ = ["DilatedResBlock", "PerCellProfileNetWide", "BiasNet", "multinomial_nll", "count_mse"]
