"""Optional one-gene integration test for chained ABC preprocessing.

Set ``EPINFORMER_REAL_DATA_ROOT`` to a reproduction workspace containing the
K562 ABC outputs and shared data directory.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_abc_pipeline import _chain_preprocessing


@unittest.skipUnless(os.environ.get("EPINFORMER_REAL_DATA_ROOT"),
                     "set EPINFORMER_REAL_DATA_ROOT to run")
class RealDataChainTest(unittest.TestCase):
    def test_one_gene_creates_factored_hdf5(self):
        root = Path(os.environ["EPINFORMER_REAL_DATA_ROOT"])
        links = root / "batch_output" / "K562" / "links"
        pred = pd.read_csv(
            links / "Predictions" / "EnhancerPredictionsAllPutative.txt", sep="\t"
        )
        enh = pd.read_csv(links / "EnhancerList.txt", sep="\t")
        expr_path = root / "data" / "GM12878_K562_18377_gene_expr_fromXpresso.csv"
        expr = pd.read_csv(expr_path)

        usable = pred[
            pred["TargetGene"].isin(set(expr["ENSID"]))
            & pred["distance"].between(1_001, 100_000)
        ]
        ensid = usable["TargetGene"].iloc[0]
        pred_one = pred[(pred["TargetGene"] == ensid) & (pred["distance"] <= 100_000)]
        enh_one = enh[enh["name"].isin(set(pred_one["name"]))]
        expr_one = expr[expr["ENSID"] == ensid]

        with tempfile.TemporaryDirectory(prefix="epinformer_chain_") as tmp:
            tmp_path = Path(tmp)
            pred_path = tmp_path / "predictions.tsv"
            enh_path = tmp_path / "enhancers.tsv"
            expr_one_path = tmp_path / "expression.csv"
            pred_one.to_csv(pred_path, sep="\t", index=False)
            enh_one.to_csv(enh_path, sep="\t", index=False)
            expr_one.to_csv(expr_one_path, index=False)

            output = tmp_path / "encoded"
            args = SimpleNamespace(
                preprocessing_output_dir=str(output),
                output_dir=tmp,
                fasta=str(root / "data" / "reference" / "hg38" / "hg38.fa"),
                expression=str(expr_one_path),
                preset=None,
                preprocessing_min_distance=0,
                preprocessing_max_distance=100_000,
                preprocessing_n_enhancer=60,
                preprocessing_max_seq_len=2000,
                preprocessing_tss_column="TSS_xpresso",
                preprocessing_include_self_promoter=False,
                cell_type="K562",
            )
            _chain_preprocessing(
                args,
                {"predictions": str(pred_path), "enhancer_list": str(enh_path)},
            )

            h5_path = output / "K562_samples.h5"
            self.assertTrue(h5_path.is_file())
            with h5py.File(h5_path, "r") as h5:
                self.assertEqual(h5["ensid"].shape, (1,))
                self.assertEqual(h5["promoter_seq"].shape, (1, 2000, 4))
                self.assertEqual(h5["gene_enh_idx"].shape, (1, 60))
                self.assertGreater(h5["enhancer_seq"].shape[0], 0)


if __name__ == "__main__":
    unittest.main()
