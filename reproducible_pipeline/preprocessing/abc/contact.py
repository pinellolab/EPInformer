"""
Contact scoring for the ABC pipeline.

Provides power-law distance-based contact estimation and optional Hi-C
lookup via the hic-straw library.  When Hi-C data is unavailable (or the
library is not installed), all functions fall back to the power-law model.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional hicstraw import
# ---------------------------------------------------------------------------

try:
    import hicstraw

    _HAS_HICSTRAW = True
except ImportError:
    _HAS_HICSTRAW = False


# ---------------------------------------------------------------------------
# 1. Power-law contact
# ---------------------------------------------------------------------------

# Sentinel returned for distance == 0 (self-contact / same-bin).
_MAX_CONTACT: float = 1e8


def powerlaw_contact(distance: float, gamma: float = 0.87) -> float:
    """Estimate contact frequency from genomic distance using a power law.

    .. deprecated:: Use :func:`get_powerlaw_at_distance` for Broad-compatible
       power-law estimation.

    Parameters
    ----------
    distance : float
        Absolute genomic distance in base pairs.
    gamma : float
        Power-law exponent (default 0.87, from Fulco et al. 2019).

    Returns
    -------
    float
        ``distance ** (-gamma)`` when *distance* > 0, or a large sentinel
        value (``1e8``) when *distance* is zero.
    """
    if distance <= 0:
        return _MAX_CONTACT
    return float(distance) ** (-gamma)


def get_powerlaw_at_distance(
    distances,
    gamma: float,
    scale: float,
    min_distance: int = 5000,
):
    """Broad ABC-compatible power-law contact estimation.

    Computes ``exp(scale + -gamma * log(distance + 1))`` with distances
    clipped to a minimum of *min_distance* (default 5 kb).

    Parameters
    ----------
    distances : float or array-like
        Genomic distances in base pairs.
    gamma : float
        Power-law exponent (fitted from Hi-C, e.g. 1.024).
    scale : float
        Intercept / scale parameter (fitted, e.g. 5.96).
    min_distance : int
        Distances below this value are clipped up (default 5000).

    Returns
    -------
    numpy.ndarray or float
        Estimated contact frequency.
    """
    assert gamma > 0
    assert scale > 0
    distances = np.clip(distances, min_distance, np.inf)
    log_dists = np.log(np.asarray(distances, dtype=float) + 1)
    return np.exp(scale + (-gamma) * log_dists)


# ---------------------------------------------------------------------------
# 2. Hi-C data loader
# ---------------------------------------------------------------------------

class HiCContactMap:
    """Wrapper around a ``.hic`` file matching the Broad ABC pipeline.

    Uses ``hicstraw.HiCFile`` + ``getMatrixZoomData`` with **SCALE**
    normalization (fallback: KR → VC → NONE).  The full chromosome matrix
    is fetched in chunks, doubly-stochastic normalised (divide all values
    by the mean row/column sum), and diagonal bins are corrected.

    Parameters
    ----------
    hic_file : str
        Path to a ``.hic`` file readable by *hicstraw*.
    resolution : int
        Bin resolution in base pairs (default 5000).
    tss_hic_contribution : float
        Scaling factor for diagonal correction (default 100).
    """

    _NORM_FALLBACK = ["SCALE", "KR", "VC", "NONE"]

    def __init__(
        self,
        hic_file: str,
        resolution: int = 5000,
        tss_hic_contribution: float = 100.0,
    ) -> None:
        if not _HAS_HICSTRAW:
            raise ImportError(
                "The 'hic-straw' package is required for Hi-C contact lookup. "
                "Install it with:  pip install hic-straw"
            )
        self.hic_file_path = hic_file
        self.resolution = resolution
        self.tss_hic_contribution = tss_hic_contribution
        self._hic: Optional[object] = None  # hicstraw.HiCFile
        self._chrom_style: Optional[str] = None
        # Cache: chrom -> pd.DataFrame with columns [bin1, bin2, hic_contact]
        # (upper-triangle, bin indices in resolution units)
        self._cache: Dict[str, pd.DataFrame] = {}

    # -- internal helpers ----------------------------------------------------

    def _open(self) -> None:
        if self._hic is not None:
            return
        self._hic = hicstraw.HiCFile(self.hic_file_path)
        names = [c.name for c in self._hic.getChromosomes()]
        # index 0 is often "All"; check index 1
        if len(names) > 1 and names[1].startswith("chr"):
            self._chrom_style = "chr"
        else:
            self._chrom_style = ""

    def _chrom_key(self, chrom: str) -> str:
        if self._chrom_style == "chr":
            return chrom if chrom.startswith("chr") else f"chr{chrom}"
        return chrom.replace("chr", "")

    def _get_chrom_size(self, chrom_key: str) -> int:
        for c in self._hic.getChromosomes():
            if c.name == chrom_key:
                return c.length
        raise RuntimeError(f"Chromosome '{chrom_key}' not found in .hic file")

    @staticmethod
    def _fill_diagonals(df, hic_resolution):
        """Replace diagonal entries with max of neighboring off-diag bins."""
        diag_mask = df.index.get_level_values("binX") == df.index.get_level_values("binY")
        diagonal_bins = df[diag_mask]
        search_space = max(1, math.ceil(5000 / hic_resolution))
        for (binX, _), _ in diagonal_bins.iterrows():
            max_contact = 0.0
            for i in range(1, search_space + 1):
                for nbr in [(binX - i, binX), (binX, binX + i)]:
                    if nbr in df.index:
                        max_contact = max(max_contact, df.loc[nbr, "counts"])
            df.loc[(binX, binX), "counts"] = max_contact

    def _ensure_chrom(self, chrom: str) -> None:
        """Fetch, normalise, and cache Hi-C data for one chromosome."""
        if chrom in self._cache:
            return

        self._open()
        chrom_key = self._chrom_key(chrom)

        try:
            chrom_size = self._get_chrom_size(chrom_key)
        except RuntimeError:
            warnings.warn(
                f"Chromosome '{chrom}' not in Hi-C file. Power-law fallback."
            )
            self._cache[chrom] = pd.DataFrame(columns=["bin1", "bin2", "hic_contact"])
            return

        # Try normalizations in order
        matrix_object = None
        used_norm = None
        for norm in self._NORM_FALLBACK:
            try:
                matrix_object = self._hic.getMatrixZoomData(
                    chrom_key, chrom_key, "observed", norm, "BP", self.resolution,
                )
                used_norm = norm
                break
            except Exception:
                continue

        if matrix_object is None:
            warnings.warn(
                f"No working normalization for '{chrom}'. Power-law fallback."
            )
            self._cache[chrom] = pd.DataFrame(columns=["bin1", "bin2", "hic_contact"])
            return

        if chrom not in self._cache:  # first chromosome: announce
            print(f"  Hi-C: using {used_norm} normalization at {self.resolution}bp.")

        # Fetch in chunks (8000-row blocks), accumulate bin_sums
        num_rows = 8000
        step_size = num_rows * self.resolution
        bin_sums: Dict[int, float] = defaultdict(float)
        all_records = []

        for start in range(0, chrom_size, step_size):
            end = start + step_size - self.resolution
            recs = matrix_object.getRecords(start, end, 0, chrom_size)
            if not recs:
                continue
            records = [[r.binX, r.binY, r.counts] for r in recs]
            # Accumulate bin sums (handle query-boundary duplicates)
            for binX, binY, value in records:
                adj_value = value
                if binX < start or binY > end:
                    adj_value = value / 2
                if binX == binY:
                    bin_sums[binX] += adj_value
                else:
                    bin_sums[binX] += adj_value
                    bin_sums[binY] += adj_value
            all_records.extend(records)

        if not all_records:
            self._cache[chrom] = pd.DataFrame(columns=["bin1", "bin2", "hic_contact"])
            return

        # Build DataFrame, normalise bins, fill diagonals
        df = pd.DataFrame(all_records, columns=["binX", "binY", "counts"])
        df["binX"] = np.floor(df["binX"] / self.resolution).astype(int)
        df["binY"] = np.floor(df["binY"] / self.resolution).astype(int)
        df = df.set_index(["binX", "binY"])
        # De-duplicate (chunks may overlap at boundaries) — keep max
        df = df.groupby(level=["binX", "binY"]).max()

        # Diagonal correction
        self._fill_diagonals(df, self.resolution)
        # Apply tss_hic_contribution scaling to diagonal
        diag_mask = df.index.get_level_values("binX") == df.index.get_level_values("binY")
        df.loc[diag_mask, "counts"] *= self.tss_hic_contribution / 100.0

        # Doubly-stochastic: divide by mean row sum
        row_mean = np.mean(list(bin_sums.values())) if bin_sums else 1.0
        if row_mean > 0 and abs(row_mean - 1) > 0.001:
            df["counts"] /= row_mean

        # Convert to upper-triangular output format
        df = df.reset_index()
        # Ensure upper triangle (binX <= binY)
        swap = df["binX"] > df["binY"]
        df.loc[swap, ["binX", "binY"]] = df.loc[swap, ["binY", "binX"]].values
        df = df.groupby(["binX", "binY"])["counts"].max().reset_index()
        df.columns = ["bin1", "bin2", "hic_contact"]

        self._cache[chrom] = df

    # -- public API ----------------------------------------------------------

    def get_hic_for_pred(self, pred: pd.DataFrame, chrom: str) -> pd.DataFrame:
        """Merge Hi-C contact into a predictions DataFrame for one chromosome.

        Parameters
        ----------
        pred : pd.DataFrame
            Must have ``element_mid`` and ``tss`` columns.
        chrom : str
            Chromosome name.

        Returns
        -------
        pd.DataFrame
            *pred* with ``hic_contact`` column added (NaN where missing).
        """
        self._ensure_chrom(chrom)
        hic_df = self._cache.get(chrom, pd.DataFrame())

        if hic_df.empty:
            pred["hic_contact"] = np.nan
            return pred

        pred = pred.copy()
        pred["enh_bin"] = np.floor(pred["element_mid"] / self.resolution).astype(int)
        pred["tss_bin"] = np.floor(pred["tss"] / self.resolution).astype(int)
        pred["bin1"] = np.minimum(pred["enh_bin"], pred["tss_bin"])
        pred["bin2"] = np.maximum(pred["enh_bin"], pred["tss_bin"])

        pred = pred.merge(hic_df, how="left", on=["bin1", "bin2"])
        pred.drop(columns=["enh_bin", "tss_bin", "bin1", "bin2"],
                  inplace=True, errors="ignore")
        return pred


def load_hic(
    hic_file: str,
    resolution: int = 5000,
    tss_hic_contribution: float = 100.0,
) -> HiCContactMap:
    """Load a Hi-C contact map from a ``.hic`` file.

    Parameters
    ----------
    hic_file : str
        Path to the ``.hic`` file.
    resolution : int
        Bin resolution in base pairs (default 5000).
    tss_hic_contribution : float
        Diagonal correction scaling (default 100).

    Returns
    -------
    HiCContactMap
        Object supporting ``.get_hic_for_pred()`` lookups.

    Raises
    ------
    ImportError
        If ``hic-straw`` is not installed.
    """
    return HiCContactMap(
        hic_file,
        resolution=resolution,
        tss_hic_contribution=tss_hic_contribution,
    )


# ---------------------------------------------------------------------------
# 2b. ABC "average Hi-C" loader (ENCODE-E2G / ABC average contact values)
# ---------------------------------------------------------------------------

class AverageHiCContactMap:
    """ABC *average* Hi-C contact map (e.g. ENCODE ``ENCFF134PUN``, GRCh38, 5 kb).

    The ABC / ENCODE-rE2G "average Hi-C" (``--hic_type avg`` in the Broad
    pipeline) is a per-chromosome table of *intra*-chromosomal averaged contact
    values.  Each per-chromosome file is a headerless TSV whose first three
    columns are ``x1``, ``x2`` (bin **start** positions, base pairs) and
    ``hic_contact`` — exactly the columns Broad's ``load_hic_avg`` reads.  The
    values are already KR-normalised + power-law-processed, so — unlike the
    ``.hic`` (juicebox) path — **no** doubly-stochastic normalisation or diagonal
    fill is applied here; the values are used as-is and then power-law-scaled +
    pseudocounted downstream in :func:`get_contacts_for_pairs`.

    The single 58 GB ENCODE file is split into this per-chromosome layout once by
    ``scripts/split_avg_hic.py`` (streaming, low memory).  Pointing this class at
    the *unsplit* single file is refused — it would not fit in memory.

    Exposes the same :meth:`get_hic_for_pred` interface as :class:`HiCContactMap`
    so it drops into :func:`get_contacts_for_pairs` with no other changes.

    Parameters
    ----------
    avg_hic_dir : str
        Directory of per-chromosome files (``chr1.tsv.gz`` etc.).
    resolution : int
        Bin resolution in base pairs (default 5000 — the ABC average-Hi-C res).
    """

    # Per-chromosome filename patterns tried, in order ({chrom} -> e.g. "chr1").
    _PATTERNS = [
        "{chrom}.tsv.gz", "{chrom}.bed.gz", "{chrom}.txt.gz", "{chrom}.avg.gz",
        "{chrom}.gz", "{chrom}.KRnorm.gz", "{chrom}/{chrom}.tsv.gz",
        "{chrom}/{chrom}.KRobserved.gz", "{chrom}/{chrom}.gz",
    ]

    def __init__(self, avg_hic_dir: str, resolution: int = 5000) -> None:
        import os
        if not os.path.isdir(avg_hic_dir):
            raise NotADirectoryError(
                f"Average Hi-C source must be a directory of per-chromosome files, "
                f"got '{avg_hic_dir}'. Split the single ENCODE file first:\n"
                f"    python scripts/split_avg_hic.py --in <ENCFF134PUN.bed.gz> --out {avg_hic_dir}"
            )
        self.avg_hic_dir = avg_hic_dir
        self.resolution = resolution
        # chrom -> DataFrame[bin1, bin2, hic_contact]
        self._cache: Dict[str, pd.DataFrame] = {}

    def _find_chrom_file(self, chrom: str) -> Optional[str]:
        import os
        candidates = [chrom]
        # tolerate chr-prefix mismatch between predictions and the split files
        candidates.append(chrom[3:] if chrom.startswith("chr") else f"chr{chrom}")
        for c in candidates:
            for pat in self._PATTERNS:
                p = os.path.join(self.avg_hic_dir, pat.format(chrom=c))
                if os.path.exists(p):
                    return p
        return None

    def _ensure_chrom(self, chrom: str) -> None:
        if chrom in self._cache:
            return
        path = self._find_chrom_file(chrom)
        if path is None:
            warnings.warn(
                f"No average Hi-C file for '{chrom}' in {self.avg_hic_dir}. "
                f"Power-law fallback for this chromosome."
            )
            self._cache[chrom] = pd.DataFrame(columns=["bin1", "bin2", "hic_contact"])
            return
        # Read only the first three columns (x1, x2, hic_contact); ignore extras.
        df = pd.read_csv(
            path, sep="\t", header=None, usecols=[0, 1, 2],
            names=["x1", "x2", "hic_contact"],
            dtype={"x1": np.int64, "x2": np.int64, "hic_contact": np.float64},
        )
        b1 = np.floor(df["x1"].values / self.resolution).astype(np.int64)
        b2 = np.floor(df["x2"].values / self.resolution).astype(np.int64)
        lo = np.minimum(b1, b2)
        hi = np.maximum(b1, b2)
        out = pd.DataFrame({"bin1": lo, "bin2": hi, "hic_contact": df["hic_contact"].values})
        # collapse duplicate bin-pairs (upper triangle) — keep the max contact
        out = out.groupby(["bin1", "bin2"], as_index=False)["hic_contact"].max()
        self._cache[chrom] = out

    def get_hic_for_pred(self, pred: pd.DataFrame, chrom: str) -> pd.DataFrame:
        """Merge average Hi-C contact into a predictions DataFrame for one chromosome.

        Same contract as :meth:`HiCContactMap.get_hic_for_pred`: *pred* must have
        ``element_mid`` and ``tss`` columns; returns *pred* with a ``hic_contact``
        column (NaN where the bin-pair is absent).
        """
        self._ensure_chrom(chrom)
        hic_df = self._cache.get(chrom, pd.DataFrame())

        if hic_df.empty:
            pred = pred.copy()
            pred["hic_contact"] = np.nan
            return pred

        pred = pred.copy()
        pred["enh_bin"] = np.floor(pred["element_mid"] / self.resolution).astype(int)
        pred["tss_bin"] = np.floor(pred["tss"] / self.resolution).astype(int)
        pred["bin1"] = np.minimum(pred["enh_bin"], pred["tss_bin"])
        pred["bin2"] = np.maximum(pred["enh_bin"], pred["tss_bin"])

        pred = pred.merge(hic_df, how="left", on=["bin1", "bin2"])
        pred.drop(columns=["enh_bin", "tss_bin", "bin1", "bin2"],
                  inplace=True, errors="ignore")
        return pred


def load_avg_hic(avg_hic_dir: str, resolution: int = 5000) -> AverageHiCContactMap:
    """Load an ABC *average* Hi-C contact map from a per-chromosome directory.

    See :class:`AverageHiCContactMap`.  Produce the directory from the single
    ENCODE file (``ENCFF134PUN``) with ``scripts/split_avg_hic.py``.
    """
    return AverageHiCContactMap(avg_hic_dir, resolution=resolution)


# ---------------------------------------------------------------------------
# 3. QC — replace genes with insufficient Hi-C coverage
# ---------------------------------------------------------------------------

def qc_hic(
    pred: pd.DataFrame,
    gamma: float,
    scale: float,
    resolution: int,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Replace Hi-C values with power-law for genes with poor Hi-C coverage.

    Genes whose self-promoter ``hic_contact`` sum is below *threshold* get
    all their ``hic_contact`` values replaced with the power-law estimate.
    This matches the Broad ABC pipeline's QC step.
    """
    if "isSelfPromoter" not in pred.columns:
        return pred

    sp = pred.loc[pred["isSelfPromoter"]]
    if sp.empty:
        return pred

    gene_key = "TargetGene" if "TargetGene" in pred.columns else "symbol"
    if gene_key not in pred.columns:
        return pred

    summ = sp.groupby(gene_key).agg({"hic_contact": "sum"})
    bad_genes = summ.loc[summ["hic_contact"] < threshold].index
    if len(bad_genes) == 0:
        return pred

    affected = pred[gene_key].isin(bad_genes)
    pred.loc[affected, "hic_contact"] = get_powerlaw_at_distance(
        pred.loc[affected, "distance"].values,
        gamma,
        scale,
        min_distance=resolution,
    )
    return pred


# ---------------------------------------------------------------------------
# 4. Vectorised batch contact scoring (Broad-compatible)
# ---------------------------------------------------------------------------

def get_contacts_for_pairs(
    pairs_df: pd.DataFrame,
    hic_data: Optional[HiCContactMap] = None,
    hic_gamma: float = 1.024238616787792,
    hic_scale: float = 5.9594510043736655,
    hic_gamma_reference: float = 0.87,
    hic_pseudocount_distance: int = 5000,
    scale_hic_using_powerlaw: bool = True,
    resolution: int = 5000,
) -> pd.DataFrame:
    """Compute contact scores for a DataFrame of TSS–element pairs.

    Follows the Broad ABC pipeline:
    1. ``powerlaw_contact`` = observed power-law (hic_gamma, hic_scale)
    2. ``powerlaw_contact_reference`` = reference power-law (hic_gamma_reference)
    3. Hi-C lookup + QC (done externally before calling this, or in predictions.py)
    4. ``hic_contact_pl_scaled`` = hic * (reference / observed)
    5. ``hic_pseudocount`` = min(powerlaw_contact, powerlaw_at_pseudocount_distance)
    6. ``hic_contact_pl_scaled_adj`` = scaled + pseudocount

    Parameters
    ----------
    pairs_df : pandas.DataFrame
        Must contain columns ``chrom``, ``tss``, ``element_mid``, ``distance``.
    hic_data : HiCContactMap, optional
        Pre-loaded Hi-C map.  ``None`` → power-law only.
    hic_gamma, hic_scale : float
        Fitted power-law parameters for the observed Hi-C.
    hic_gamma_reference : float
        Reference gamma (default 0.87).
    hic_pseudocount_distance : int
        Distance at which pseudocount is evaluated (default 5000).
    scale_hic_using_powerlaw : bool
        Whether to scale Hi-C by reference/observed power-law ratio.
    resolution : int
        Hi-C bin resolution.

    Returns
    -------
    pandas.DataFrame
        Copy of *pairs_df* with contact columns added.
    """
    df = pairs_df.copy()
    distance = df["distance"].values.astype(float)

    # --- Power-law contacts (Broad formula) --------------------------------
    df["powerlaw_contact"] = get_powerlaw_at_distance(
        distance, hic_gamma, hic_scale,
    )

    hic_scale_reference = -4.80 + 11.63 * hic_gamma_reference
    df["powerlaw_contact_reference"] = get_powerlaw_at_distance(
        distance, hic_gamma_reference, hic_scale_reference,
    )

    if hic_data is None:
        return df

    # --- Hi-C lookup, chromosome by chromosome -----------------------------
    frames = []
    for chrom, grp in df.groupby("chrom"):
        grp = hic_data.get_hic_for_pred(grp, chrom)
        frames.append(grp)
    df = pd.concat(frames, ignore_index=True)

    # Fill NaN with 0 (Broad: pred.fillna(value={"hic_contact": 0}))
    df["hic_contact"] = df["hic_contact"].fillna(0)

    # --- Power-law scaling -------------------------------------------------
    if scale_hic_using_powerlaw:
        df["hic_contact_pl_scaled"] = df["hic_contact"] * (
            df["powerlaw_contact_reference"] / df["powerlaw_contact"]
        )
    else:
        df["hic_contact_pl_scaled"] = df["hic_contact"]

    # --- Pseudocount -------------------------------------------------------
    pseudocount_at_dist = get_powerlaw_at_distance(
        hic_pseudocount_distance, hic_gamma, hic_scale, hic_pseudocount_distance,
    )
    df["hic_pseudocount"] = np.minimum(
        df["powerlaw_contact"].values, pseudocount_at_dist,
    )
    df["hic_contact_pl_scaled_adj"] = (
        df["hic_contact_pl_scaled"] + df["hic_pseudocount"]
    )

    return df
