"""Regression tests for pipeline CLI and evaluation contracts."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluate
import run_abc_pipeline
from preprocessing.hdf5 import create_pe_arrays_h5, write_pe_sample
from preprocessing import pipelines_legacy


class EvaluationFallbackTests(unittest.TestCase):
    def _write_results(self, directory: str, folds) -> None:
        pd.DataFrame(
            {
                "Pred": [float(f) for f in folds],
                "actual": [float(f) + 0.1 for f in folds],
                "fold_idx": list(folds),
            }
        ).to_csv(os.path.join(directory, "model.folds_results.csv"), index=False)

    def test_aggregate_fallback_requires_all_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_results(tmp, range(1, 12))
            with self.assertRaises(SystemExit):
                evaluate._load_expression(tmp)

    def test_aggregate_fallback_accepts_complete_cv(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_results(tmp, range(1, 13))
            frame, source = evaluate._load_expression(tmp)
            self.assertEqual(set(frame["fold"]), set(range(1, 13)))
            self.assertIn("aggregate fallback", source)


class ChainedPreprocessingTests(unittest.TestCase):
    def test_chain_uses_factored_hdf5_builder(self):
        called = mock.Mock()
        fake_module = types.ModuleType("preprocessing.pipelines_legacy")
        fake_module.obtain_PE_withSignals = called
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sys.modules, {"preprocessing.pipelines_legacy": fake_module}
        ):
            args = SimpleNamespace(
                preprocessing_output_dir=os.path.join(tmp, "encoded"),
                output_dir=tmp,
                fasta="genome.fa",
                expression="expression.csv",
                preset=None,
                preprocessing_min_distance=0,
                preprocessing_max_distance=100_000,
                preprocessing_n_enhancer=60,
                preprocessing_max_seq_len=2000,
                preprocessing_tss_column="TSS_xpresso",
                preprocessing_include_self_promoter=False,
                cell_type="K562",
            )
            run_abc_pipeline._chain_preprocessing(
                args, {"predictions": "predictions.tsv", "enhancer_list": "enhancers.tsv"}
            )

        called.assert_called_once()
        _, kwargs = called.call_args
        self.assertEqual(kwargs["max_distance"], 100_000)
        self.assertEqual(kwargs["n_enhancer"], 60)
        self.assertEqual(kwargs["signal_files"], [])
        self.assertEqual(kwargs["tss_column"], "TSS_xpresso")


class LegacyHdf5CompatibilityTests(unittest.TestCase):
    def test_legacy_writer_maps_tokens_into_factored_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.h5")
            h5 = create_pe_arrays_h5(
                path,
                n_samples=2,
                max_n_enhancer=2,
                max_seq_len=4,
                n_signal_tracks=1,
            )
            seq = np.zeros((3, 4, 4), dtype=np.float32)
            seq[0, :, 0] = 1  # promoter
            seq[1, :, 1] = 1  # one populated enhancer; slot 2 remains padding
            signal = np.arange(12, dtype=np.float32).reshape(3, 4, 1)
            write_pe_sample(
                h5,
                1,
                "ENSG_TEST",
                seq,
                activity=[2.0, 0.0],
                dhs=[3.0, 0.0],
                distance=[100.0, 0.0],
                contact=[0.5, 0.0],
                seq_signal=signal,
                n_signal_tracks=1,
            )

            self.assertEqual(h5["enhancer_seq"].shape[0], 4)
            self.assertEqual(h5["gene_enh_idx"][1].tolist(), [2, -1])
            self.assertEqual(h5["enhancer_name"][2], b"ENSG_TEST:legacy_slot_0")
            np.testing.assert_array_equal(h5["promoter_signal"][1], signal[0])
            np.testing.assert_array_equal(h5["enhancer_signal"][2], signal[1])
            h5.close()

    def test_hardcoded_legacy_drivers_fail_fast_with_migration_message(self):
        retired = [
            pipelines_legacy.obtain_GM12878_PE,
            pipelines_legacy.obtain_GM12878_H3K27ac_PE,
            pipelines_legacy.obtain_GM12878_PE_withSignals,
        ]
        for driver in retired:
            with self.subTest(driver=driver.__name__), self.assertRaisesRegex(
                RuntimeError, "run_preprocessing.py"
            ):
                driver()


if __name__ == "__main__":
    unittest.main()
