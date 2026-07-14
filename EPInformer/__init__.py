"""EPInformer model package (pipeline build).

Exposes the ``EPInformer_v2`` gene-expression model and the enhancer sequence
encoder classes from ``models.py`` (copied verbatim from the main EPInformer
repo). This is the *only* model file in this pipeline folder.
"""

from .models import (
    EPInformer_v2,
    enhancer_predictor_256bp,
    enhancer_classifier_256bp,
    seq_256bp_encoder,
)

__all__ = [
    "EPInformer_v2",
    "enhancer_predictor_256bp",
    "enhancer_classifier_256bp",
    "seq_256bp_encoder",
]
