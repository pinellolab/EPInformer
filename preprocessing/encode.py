"""DNA one-hot encoding (kipoiseq)."""

import numpy as np
import kipoiseq


def one_hot_encode(sequence, neutral_value: float = 0.0) -> np.ndarray:
    """One-hot encode a DNA string; N/ambiguous bases use ``neutral_value`` per channel."""
    return kipoiseq.transforms.functional.one_hot_dna(
        sequence, neutral_value=neutral_value
    ).astype(np.float32)
