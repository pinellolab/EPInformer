"""Tests for the original EPInformer-seq and EPInformer-seq-v2 interfaces."""

import unittest

import numpy as np
import torch

from EPInformer.epinformer_seq_v2 import (
    EPInformerSeqV2,
    EPInformerSeqV2Bias,
    INPUT_WINDOW,
    PROFILE_WINDOW,
    activity_from_profiles,
    one_hot_dna,
    reverse_complement_one_hot,
)
from EPInformer.models import enhancer_predictor_256bp
from predict_epinformer_seq import one_hot_dna as one_hot_dna_v1
from predict_epinformer_seq import predict as predict_v1


class EPInformerSeqV2Tests(unittest.TestCase):
    def test_one_hot_and_reverse_complement(self):
        encoded = one_hot_dna("ACGTN", length=5)
        self.assertEqual(encoded.shape, (4, 5))
        self.assertEqual(encoded[:, :4].sum(), 4)
        self.assertEqual(encoded[:, 4].sum(), 0)
        np.testing.assert_array_equal(
            reverse_complement_one_hot(encoded), one_hot_dna("NACGT", length=5)
        )

    def test_profile_model_shapes(self):
        model = EPInformerSeqV2(stem_channels=8, body_channels=4, n_dilated=2)
        with torch.inference_mode():
            profile, counts = model(torch.zeros(1, 4, INPUT_WINDOW))
        self.assertEqual(profile.shape, (1, 2, PROFILE_WINDOW))
        self.assertEqual(counts.shape, (1, 2))

    def test_bias_model_shapes(self):
        model = EPInformerSeqV2Bias(stem_channels=8, body_channels=4, n_dilated=2)
        with torch.inference_mode():
            profile, counts = model(torch.zeros(1, 4, PROFILE_WINDOW))
        self.assertEqual(profile.shape, (1, 2, PROFILE_WINDOW))
        self.assertEqual(counts.shape, (1, 2))

    def test_activity_aggregation_uses_central_window(self):
        dnase = np.zeros(PROFILE_WINDOW, dtype=np.float32)
        h3k27ac = np.zeros(PROFILE_WINDOW, dtype=np.float32)
        dnase[0] = h3k27ac[0] = 100.0
        dnase[PROFILE_WINDOW // 2] = 4.0
        h3k27ac[PROFILE_WINDOW // 2] = 9.0
        self.assertEqual(activity_from_profiles(dnase, h3k27ac, "Enhancer_DNase"), 4.0)
        self.assertEqual(activity_from_profiles(dnase, h3k27ac, "Enhancer_H3K27ac"), 9.0)
        self.assertTrue(
            np.isclose(
                activity_from_profiles(dnase, h3k27ac, "Enhancer_H3K27ac_DNase"),
                6.0,
            )
        )


class OriginalEPInformerSeqTests(unittest.TestCase):
    def test_prediction(self):
        encoded = one_hot_dna_v1("ACGT" * 64)
        self.assertEqual(encoded.shape, (4, 256))
        model = enhancer_predictor_256bp().eval()
        prediction_log2, activity = predict_v1(model, "ACGT" * 64, "cpu")
        self.assertIsInstance(prediction_log2, float)
        self.assertGreaterEqual(activity, 0.0)


if __name__ == "__main__":
    unittest.main()
