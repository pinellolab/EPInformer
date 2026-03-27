"""
Contact scoring for the ABC pipeline.

Provides power-law distance-based contact estimation and optional Hi-C
lookup via the hic-straw library.  When Hi-C data is unavailable (or the
library is not installed), all functions fall back to the power-law model.
"""

from __future__ import annotations

import warnings
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


# ---------------------------------------------------------------------------
# 2. Hi-C data loader
# ---------------------------------------------------------------------------

class HiCContactMap:
    """Lazy-loading wrapper around a ``.hic`` file.

    Chromosome pair data is fetched on first access and cached so that
    repeated queries to the same chromosome are fast.

    Parameters
    ----------
    hic_file : str
        Path to a ``.hic`` file readable by *hicstraw*.
    resolution : int
        Bin resolution in base pairs (default 5000).
    normalization : str
        Matrix normalization type (default ``"KR"``).
    """

    def __init__(
        self,
        hic_file: str,
        resolution: int = 5000,
        normalization: str = "KR",
    ) -> None:
        if not _HAS_HICSTRAW:
            raise ImportError(
                "The 'hic-straw' package is required for Hi-C contact lookup. "
                "Install it with:  pip install hic-straw"
            )
        self.hic_file = hic_file
        self.resolution = resolution
        self.normalization = normalization
        self._hic: Optional[hicstraw.HiCFile] = None
        # Cache: chrom -> dict[(bin_i, bin_j) -> value]
        self._cache: Dict[str, Dict[tuple, float]] = {}

    # -- internal helpers ----------------------------------------------------

    def _open(self) -> "hicstraw.HiCFile":
        if self._hic is None:
            self._hic = hicstraw.HiCFile(self.hic_file)
        return self._hic

    def _ensure_chrom(self, chrom: str) -> None:
        """Load all records for an intra-chromosomal matrix into the cache."""
        if chrom in self._cache:
            return

        hic = self._open()
        # hicstraw expects chromosome names without the "chr" prefix in
        # some builds; try both.
        chrom_key = chrom
        try:
            mzd = hic.getMatrixZoomData(
                chrom_key, chrom_key, "observed", self.normalization, "BP", self.resolution
            )
        except Exception:
            # Retry without "chr" prefix
            chrom_key = chrom.replace("chr", "")
            try:
                mzd = hic.getMatrixZoomData(
                    chrom_key, chrom_key, "observed", self.normalization, "BP", self.resolution
                )
            except Exception:
                warnings.warn(
                    f"Could not load Hi-C data for chromosome '{chrom}'. "
                    "Power-law fallback will be used for this chromosome."
                )
                self._cache[chrom] = {}
                return

        records = mzd.getRecords()
        contact_dict: Dict[tuple, float] = {}
        for rec in records:
            contact_dict[(rec.binX, rec.binY)] = rec.counts
            contact_dict[(rec.binY, rec.binX)] = rec.counts
        self._cache[chrom] = contact_dict

    # -- public API ----------------------------------------------------------

    def query(self, chrom: str, pos1: int, pos2: int) -> Optional[float]:
        """Look up the Hi-C contact value between two loci.

        Parameters
        ----------
        chrom : str
            Chromosome name (e.g. ``"chr1"``).
        pos1, pos2 : int
            Genomic positions in base pairs.

        Returns
        -------
        float or None
            The observed (normalized) contact value, or ``None`` if the
            bin pair is not present in the matrix.
        """
        self._ensure_chrom(chrom)
        bin1 = (pos1 // self.resolution) * self.resolution
        bin2 = (pos2 // self.resolution) * self.resolution
        return self._cache.get(chrom, {}).get((bin1, bin2))


def load_hic(
    hic_file: str,
    resolution: int = 5000,
    normalization: str = "KR",
) -> HiCContactMap:
    """Load a Hi-C contact map from a ``.hic`` file.

    Parameters
    ----------
    hic_file : str
        Path to the ``.hic`` file.
    resolution : int
        Bin resolution in base pairs (default 5000).
    normalization : str
        Normalization method (default ``"KR"``).

    Returns
    -------
    HiCContactMap
        Object supporting ``.query(chrom, pos1, pos2)`` lookups.  Chromosome
        data is loaded lazily and cached.

    Raises
    ------
    ImportError
        If ``hic-straw`` is not installed.
    """
    return HiCContactMap(hic_file, resolution=resolution, normalization=normalization)


# ---------------------------------------------------------------------------
# 3. Single-pair contact lookup
# ---------------------------------------------------------------------------

def get_contact(
    gene_tss: int,
    element_mid: int,
    chrom: str,
    hic_data: Optional[HiCContactMap] = None,
    gamma: float = 0.87,
    resolution: int = 5000,
) -> float:
    """Return a contact score between a TSS and a candidate element.

    When *hic_data* is provided, the Hi-C observed value is looked up and
    scaled by the power-law expectation so that the score is comparable
    across distances (``hic_contact * distance ** gamma``).  If the Hi-C
    value is missing or zero, the power-law estimate is used as a fallback.

    Parameters
    ----------
    gene_tss : int
        TSS position (bp).
    element_mid : int
        Element midpoint position (bp).
    chrom : str
        Chromosome name.
    hic_data : HiCContactMap, optional
        Pre-loaded Hi-C map (from :func:`load_hic`).  ``None`` means
        power-law only.
    gamma : float
        Power-law exponent.
    resolution : int
        Hi-C bin resolution (only used for the lookup).

    Returns
    -------
    float
        Contact score.
    """
    distance = abs(gene_tss - element_mid)
    pl_contact = powerlaw_contact(distance, gamma=gamma)

    if hic_data is None:
        return pl_contact

    hic_val = hic_data.query(chrom, gene_tss, element_mid)

    # Fallback when Hi-C is missing / NaN / zero
    if hic_val is None or np.isnan(hic_val) or hic_val == 0:
        return pl_contact

    # Power-law-scaled Hi-C: normalises out the distance dependence so
    # that the score reflects enrichment over the expected contact.
    if distance > 0:
        hic_contact_pl_scaled = hic_val * (distance ** gamma)
    else:
        hic_contact_pl_scaled = hic_val

    return hic_contact_pl_scaled


# ---------------------------------------------------------------------------
# 4. Vectorised batch lookup
# ---------------------------------------------------------------------------

def get_contacts_for_pairs(
    pairs_df: pd.DataFrame,
    hic_data: Optional[HiCContactMap] = None,
    gamma: float = 0.87,
    resolution: int = 5000,
) -> pd.DataFrame:
    """Compute contact scores for a DataFrame of TSS–element pairs.

    Parameters
    ----------
    pairs_df : pandas.DataFrame
        Must contain columns ``chrom``, ``tss``, ``element_mid``.
    hic_data : HiCContactMap, optional
        Pre-loaded Hi-C map.  ``None`` → power-law only.
    gamma : float
        Power-law exponent.
    resolution : int
        Hi-C bin resolution.

    Returns
    -------
    pandas.DataFrame
        Copy of *pairs_df* with added columns:

        * ``powerlaw_contact`` — always present.
        * ``hic_contact`` — raw Hi-C value (only when *hic_data* given).
        * ``hic_contact_pl_scaled`` — Hi-C value scaled by distance ** gamma
          (only when *hic_data* given).  Falls back to ``powerlaw_contact``
          for missing / zero Hi-C entries.
    """
    df = pairs_df.copy()

    distance = (df["tss"] - df["element_mid"]).abs()

    # Power-law contact (vectorised)
    pl = np.where(distance > 0, distance.astype(float) ** (-gamma), _MAX_CONTACT)
    df["powerlaw_contact"] = pl

    if hic_data is None:
        return df

    # --- Hi-C lookup, chromosome by chromosome for cache efficiency ---------
    hic_vals = np.full(len(df), np.nan)
    for chrom, idx in df.groupby("chrom").groups.items():
        sub = df.loc[idx]
        for i, (tss, emid) in zip(
            idx, zip(sub["tss"], sub["element_mid"])
        ):
            val = hic_data.query(chrom, int(tss), int(emid))
            if val is not None:
                hic_vals[df.index.get_loc(i)] = val

    df["hic_contact"] = hic_vals

    # Scaled Hi-C, with power-law fallback for missing values
    hic_arr = df["hic_contact"].values.copy()
    dist_arr = distance.values.astype(float)

    valid = np.isfinite(hic_arr) & (hic_arr != 0)
    scaled = np.where(dist_arr > 0, hic_arr * (dist_arr ** gamma), hic_arr)

    # Fallback: use power-law where Hi-C is not usable
    df["hic_contact_pl_scaled"] = np.where(valid, scaled, pl)

    return df
