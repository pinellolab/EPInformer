"""
Candidate element definition for the ABC pipeline (Step 1).

Calls MACS2 for peak calling on accessibility data (DNase-seq or ATAC-seq),
selects top peaks by signal, resizes around summits, merges overlapping
regions, and optionally removes blacklisted regions.
"""

import os
import subprocess

import numpy as np
import pandas as pd
import pybedtools


# ---------------------------------------------------------------------------
# narrowPeak column specification
# ---------------------------------------------------------------------------

_NARROWPEAK_COLUMNS = [
    "chr", "start", "end", "name", "score", "strand",
    "signalValue", "pValue", "qValue", "peak",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_select_peaks(
    narrowpeak_path: str,
    logger,
    n_top_peaks: int = 150_000,
) -> pd.DataFrame:
    """Read a narrowPeak file, sort by signal, and return top peaks.

    Parameters
    ----------
    narrowpeak_path : str
        Path to a MACS2 narrowPeak file.
    logger : StepLogger
        Logger instance for progress messages.
    n_top_peaks : int
        Maximum number of peaks to retain (ranked by signalValue).

    Returns
    -------
    pandas.DataFrame
        Top peaks with standard narrowPeak columns.
    """
    peaks = pd.read_csv(
        narrowpeak_path,
        sep="\t",
        header=None,
        names=_NARROWPEAK_COLUMNS,
        comment="#",
    )

    n_total = len(peaks)
    logger.info(f"Found {n_total} peaks")

    peaks = peaks.sort_values("signalValue", ascending=False).reset_index(drop=True)

    n_select = min(n_top_peaks, n_total)
    logger.info(f"Selecting top {n_select} peaks by signal value ...")
    peaks = peaks.iloc[:n_select].copy()

    return peaks


def _resize_around_summits(
    peaks: pd.DataFrame,
    logger,
    peak_extend: int = 250,
) -> pd.DataFrame:
    """Compute summit positions and resize peaks to fixed-width windows.

    Parameters
    ----------
    peaks : pandas.DataFrame
        Must contain ``start`` and ``peak`` (summit offset) columns.
    logger : StepLogger
        Logger instance.
    peak_extend : int
        Number of bases to extend on each side of the summit.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: chr, start, end, name, summit.
    """
    window_size = peak_extend * 2
    logger.info(f"Resizing to {window_size}bp windows centered on summits ...")

    summits = peaks["start"] + peaks["peak"]

    df = pd.DataFrame({
        "chr": peaks["chr"],
        "start": (summits - peak_extend).clip(lower=0),
        "end": summits + peak_extend,
        "name": peaks["name"],
        "summit": summits,
    })

    return df


def _merge_and_filter(
    resized: pd.DataFrame,
    output_dir: str,
    logger,
    blacklist: str = None,
) -> str:
    """Sort, merge overlapping regions, remove blacklist, and save outputs.

    Parameters
    ----------
    resized : pandas.DataFrame
        Resized candidate regions with columns: chr, start, end, name, summit.
    output_dir : str
        Directory to write output BED files.
    logger : StepLogger
        Logger instance.
    blacklist : str, optional
        Path to a blacklist BED file.  Regions overlapping the blacklist
        are subtracted from the merged candidates.

    Returns
    -------
    str
        Path to the merged ``candidates.bed`` file.
    """
    # Save pre-merge candidates with summit information
    summits_path = os.path.join(output_dir, "candidates_with_summits.bed")
    resized.to_csv(summits_path, sep="\t", header=False, index=False)
    logger.info(f"Saved: {summits_path}")

    # Sort and merge overlapping regions
    logger.info("Merging overlapping regions ...")
    bed = pybedtools.BedTool.from_dataframe(
        resized[["chr", "start", "end"]]
    ).sort().merge()

    # Remove blacklisted regions
    if blacklist is not None:
        blacklist_bed = pybedtools.BedTool(blacklist)
        bed = bed.subtract(blacklist_bed)
        logger.info(f"Removed blacklisted regions using {blacklist}")

    # Convert back to DataFrame to count and save
    candidates = bed.to_dataframe(names=["chr", "start", "end"])
    n_candidates = len(candidates)
    logger.info(f"Result: {n_candidates} candidate elements")

    candidates_path = os.path.join(output_dir, "candidates.bed")
    candidates.to_csv(candidates_path, sep="\t", header=False, index=False)
    logger.info(f"Saved: {candidates_path}")

    # Clean up pybedtools temp files
    pybedtools.cleanup()

    return candidates_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def define_candidates(
    accessibility_bam: str,
    output_dir: str,
    logger,
    assay: str = "dnase",
    n_top_peaks: int = 150_000,
    peak_extend: int = 250,
    blacklist: str = None,
) -> str:
    """Run MACS2 peak calling and define candidate regulatory elements.

    This implements Step 1 of the ABC pipeline: peak calling on an
    accessibility BAM file, selection of top peaks by signal value,
    resizing around summits, merging overlapping regions, and optional
    blacklist subtraction.

    Parameters
    ----------
    accessibility_bam : str
        Path to an indexed BAM file (DNase-seq or ATAC-seq).
    output_dir : str
        Directory for all output files.  A ``macs2/`` subdirectory is
        created for raw MACS2 output.
    logger : utils.StepLogger
        Logger instance for progress messages.
    assay : str
        Either ``"dnase"`` or ``"atac"``.  Controls MACS2 shift/extsize
        parameters.
    n_top_peaks : int
        Maximum number of peaks to retain (default 150,000).
    peak_extend : int
        Bases to extend on each side of the summit (default 250 -> 500bp
        windows).
    blacklist : str, optional
        Path to a blacklist BED file.  Overlapping regions are removed.

    Returns
    -------
    str
        Path to ``{output_dir}/candidates.bed`` (merged candidate regions).

    Raises
    ------
    ValueError
        If *assay* is not ``"dnase"`` or ``"atac"``.
    RuntimeError
        If the MACS2 subprocess returns a non-zero exit code.
    """
    assay = assay.lower()
    if assay not in ("dnase", "atac"):
        raise ValueError(f"assay must be 'dnase' or 'atac', got '{assay}'")

    os.makedirs(output_dir, exist_ok=True)
    macs2_dir = os.path.join(output_dir, "macs2")
    os.makedirs(macs2_dir, exist_ok=True)

    # ---- 1. MACS2 peak calling ---------------------------------------------
    logger.info(f"Running MACS2 peak calling ({assay} mode) ...")

    if assay == "dnase":
        shift, extsize = -75, 150
    else:  # atac
        shift, extsize = -100, 200

    macs2_cmd = [
        "macs2", "callpeak",
        "-t", accessibility_bam,
        "-f", "BAM",
        "-g", "hs",
        "--nomodel",
        "--shift", str(shift),
        "--extsize", str(extsize),
        "-B", "--SPMR",
        "--keep-dup", "all",
        "--call-summits",
        "-n", "peaks",
        "--outdir", macs2_dir,
    ]

    logger.info(f"Command: {' '.join(macs2_cmd)}")

    result = subprocess.run(
        macs2_cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"MACS2 failed with return code {result.returncode}.\n"
            f"stderr:\n{result.stderr}"
        )

    # ---- 2. Load and select top peaks --------------------------------------
    narrowpeak_path = os.path.join(macs2_dir, "peaks_peaks.narrowPeak")
    peaks = _load_and_select_peaks(narrowpeak_path, logger, n_top_peaks)

    # ---- 3. Resize around summits ------------------------------------------
    resized = _resize_around_summits(peaks, logger, peak_extend)

    # ---- 4-6. Merge, filter blacklist, save --------------------------------
    candidates_path = _merge_and_filter(resized, output_dir, logger, blacklist)

    return candidates_path


def load_candidates_from_peaks(
    peaks_file: str,
    output_dir: str,
    logger,
    n_top_peaks: int = 150_000,
    peak_extend: int = 250,
    blacklist: str = None,
) -> str:
    """Define candidate elements from an existing narrowPeak file.

    Skips the MACS2 step and directly processes a pre-computed narrowPeak
    file through peak selection, summit resizing, merging, and optional
    blacklist removal.  Intended for the ``from-peaks`` subcommand.

    Parameters
    ----------
    peaks_file : str
        Path to a narrowPeak file (10-column, tab-separated).
    output_dir : str
        Directory for output files.
    logger : utils.StepLogger
        Logger instance for progress messages.
    n_top_peaks : int
        Maximum number of peaks to retain (default 150,000).
    peak_extend : int
        Bases to extend on each side of the summit (default 250).
    blacklist : str, optional
        Path to a blacklist BED file.

    Returns
    -------
    str
        Path to ``{output_dir}/candidates.bed`` (merged candidate regions).
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading peaks from {peaks_file} ...")

    # ---- 2. Load and select top peaks --------------------------------------
    peaks = _load_and_select_peaks(peaks_file, logger, n_top_peaks)

    # ---- 3. Resize around summits ------------------------------------------
    resized = _resize_around_summits(peaks, logger, peak_extend)

    # ---- 4-6. Merge, filter blacklist, save --------------------------------
    candidates_path = _merge_and_filter(resized, output_dir, logger, blacklist)

    return candidates_path
