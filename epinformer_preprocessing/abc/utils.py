"""
Utility functions for the streamlined ABC pipeline in EPInformer.

Provides progress logging, dependency checking, read counting,
quantile normalization, and gene BED loading.
"""

import os
import time
import shutil
import subprocess

import numpy as np
import pandas as pd
import pysam


# ---------------------------------------------------------------------------
# 1. StepLogger
# ---------------------------------------------------------------------------

class StepLogger:
    """Progress reporter with step numbering and timing."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.pipeline_start = time.time()
        self._step_start = None
        self._step_name = None

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format elapsed seconds as 'Xm Ys' or 'Xs'."""
        seconds = int(round(seconds))
        if seconds >= 60:
            m, s = divmod(seconds, 60)
            return f"{m}m {s}s"
        return f"{seconds}s"

    # -- public API ----------------------------------------------------------

    def start_step(self, name: str) -> None:
        """Begin a new step, printing its header."""
        self.current_step += 1
        self._step_name = name
        self._step_start = time.time()
        print(f"[Step {self.current_step}/{self.total_steps}] {name} ...")

    def info(self, msg: str) -> None:
        """Print an indented informational message."""
        print(f"  {msg}")

    def done(self) -> None:
        """Print elapsed time for the current step."""
        if self._step_start is not None:
            elapsed = time.time() - self._step_start
            print(f"  Done ({self._fmt_time(elapsed)})")

    def summary(self, output_files: dict) -> None:
        """Print a final pipeline summary with file counts.

        Parameters
        ----------
        output_files : dict
            Mapping of description strings to file paths (or lists of paths).
        """
        total_elapsed = time.time() - self.pipeline_start
        print()
        print("=" * 60)
        print(f"Pipeline complete in {self._fmt_time(total_elapsed)}")
        print("-" * 60)
        for desc, paths in output_files.items():
            if isinstance(paths, (list, tuple)):
                print(f"  {desc}: {len(paths)} files")
            else:
                size = ""
                if isinstance(paths, str) and os.path.isfile(paths):
                    nbytes = os.path.getsize(paths)
                    if nbytes > 1_000_000:
                        size = f" ({nbytes / 1_000_000:.1f} MB)"
                    elif nbytes > 1_000:
                        size = f" ({nbytes / 1_000:.1f} KB)"
                print(f"  {desc}: {paths}{size}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# 2. check_dependencies
# ---------------------------------------------------------------------------

_INSTALL_HINTS = {
    "macs2": (
        "Install via conda:  conda install -c bioconda macs2\n"
        "  Or via pip:         pip install macs2"
    ),
    "bedtools": (
        "Install via conda:  conda install -c bioconda bedtools\n"
        "  Or see:             https://bedtools.readthedocs.io/en/latest/content/installation.html"
    ),
    "samtools": (
        "Install via conda:  conda install -c bioconda samtools\n"
        "  Or see:             http://www.htslib.org/download/"
    ),
}


def check_dependencies() -> None:
    """Verify that required command-line tools are on PATH.

    Required: macs2, bedtools, samtools.
    Optional: hic-straw Python package (warns but does not fail).

    Raises
    ------
    SystemExit
        If any required tool is missing.
    """
    missing = []
    for tool in ("macs2", "bedtools", "samtools"):
        if shutil.which(tool) is None:
            missing.append(tool)
            print(f"[ERROR] Required tool '{tool}' not found on PATH.")
            print(f"  {_INSTALL_HINTS[tool]}")

    # Optional: hic-straw
    try:
        import straw  # noqa: F401
    except ImportError:
        print(
            "[WARNING] Optional Python package 'hic-straw' is not installed.\n"
            "  Hi-C contact features will not be available.\n"
            "  Install via pip:  pip install hic-straw"
        )

    if missing:
        raise SystemExit(
            f"Missing required dependencies: {', '.join(missing)}. "
            "Please install them and retry."
        )


# ---------------------------------------------------------------------------
# 3. count_reads_in_regions
# ---------------------------------------------------------------------------

def count_reads_in_regions(
    bam_path: str,
    regions_bed: pd.DataFrame,
    chrom_sizes: dict = None,
) -> pd.DataFrame:
    """Count aligned reads overlapping BED regions using pysam.

    Parameters
    ----------
    bam_path : str
        Path to an indexed BAM file.
    regions_bed : pandas.DataFrame
        Must contain columns ``chr``, ``start``, ``end``.
    chrom_sizes : dict, optional
        Mapping of chrom name to length. Not currently used but reserved
        for future clipping of regions to chromosome bounds.

    Returns
    -------
    pandas.DataFrame
        Copy of *regions_bed* with added columns:
        ``readCount``, ``RPM``, ``RPKM``.
    """
    df = regions_bed.copy()

    bam = pysam.AlignmentFile(bam_path, "rb")
    total_mapped = bam.mapped

    counts = np.zeros(len(df), dtype=np.int64)
    for i, (chrom, start, end) in enumerate(
        zip(df["chr"], df["start"], df["end"])
    ):
        try:
            counts[i] = bam.count(contig=str(chrom), start=int(start), end=int(end))
        except ValueError:
            # Chromosome not in BAM — leave count as 0.
            counts[i] = 0

    bam.close()

    df["readCount"] = counts

    # RPM: reads per million mapped reads
    rpm_scale = total_mapped / 1e6 if total_mapped > 0 else 1.0
    df["RPM"] = df["readCount"] / rpm_scale

    # RPKM: RPM per kilobase of region
    region_length_kb = (df["end"] - df["start"]).astype(float) / 1000.0
    region_length_kb = region_length_kb.replace(0, np.nan)
    df["RPKM"] = df["RPM"] / region_length_kb

    return df


# ---------------------------------------------------------------------------
# 4. quantile_normalize
# ---------------------------------------------------------------------------

def quantile_normalize(
    values: np.ndarray,
    reference: np.ndarray = None,
) -> np.ndarray:
    """Rank-based quantile normalization.

    Parameters
    ----------
    values : numpy.ndarray
        1-D array of values to normalize.
    reference : numpy.ndarray, optional
        Sorted reference distribution. If provided, ranks in *values* are
        mapped to the corresponding quantiles of *reference*.
        If ``None``, standard quantile normalization is applied: each value
        is replaced by the mean of all values sharing its rank.

    Returns
    -------
    numpy.ndarray
        Normalized array with the same shape as *values*.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return values.copy()

    # Compute ranks (0-based, average ties)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(n, dtype=float)

    # Handle ties: assign the average rank to tied values
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    if reference is not None:
        ref_sorted = np.sort(reference)
        m = len(ref_sorted)
        # Map ranks (0..n-1) into indices of reference (0..m-1)
        ref_indices = ranks / (n - 1) * (m - 1) if n > 1 else np.zeros_like(ranks)
        # Linear interpolation into reference distribution
        result = np.interp(ref_indices, np.arange(m), ref_sorted)
    else:
        # Standard: replace each rank with mean of values at that rank
        sorted_vals = values[order].copy()
        rank_means = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j < n and sorted_vals[j] == sorted_vals[i]:
                j += 1
            mean_val = np.mean(sorted_vals[i:j])
            rank_means[i:j] = mean_val
            i = j
        result = np.empty(n, dtype=float)
        result[order] = rank_means

    return result


# ---------------------------------------------------------------------------
# 5. load_gene_bed
# ---------------------------------------------------------------------------

_GENE_BED_COLUMNS = [
    "chr", "start", "end", "symbol", "score", "strand",
    "ENSID", "gene_type",
]


def load_gene_bed(
    gene_bed_path: str,
    expression_csv: str = None,
    expression_column: str = None,
    tss_column: str = "TSS_xpresso",
) -> pd.DataFrame:
    """Load a CollapsedGeneBounds BED file and optionally merge expression.

    Parameters
    ----------
    gene_bed_path : str
        Path to the BED file (tab-separated, 8 columns).
    expression_csv : str, optional
        Path to a CSV with gene expression values.
    expression_column : str, optional
        Column name in *expression_csv* containing expression values.
        Required if *expression_csv* is provided.
    tss_column : str
        Column in *expression_csv* to use for TSS override (default
        ``"TSS_xpresso"``). If the column exists, it will replace the
        computed TSS.

    Returns
    -------
    pandas.DataFrame
        Columns: chr, start, end, symbol, ENSID, strand, gene_type, tss,
        and optionally Expression.
    """
    df = pd.read_csv(
        gene_bed_path,
        sep="\t",
        header=None,
        names=_GENE_BED_COLUMNS,
        comment="#",
    )

    # Strand-aware TSS
    df["tss"] = np.where(df["strand"] == "+", df["start"], df["end"])

    # Strip ENSID version suffix (e.g. ENSG00000177951.6 → ENSG00000177951)
    df["ENSID"] = df["ENSID"].astype(str).str.replace(r"\.\d+$", "", regex=True)

    if expression_csv is not None:
        if expression_column is None:
            raise ValueError(
                "expression_column must be specified when expression_csv is provided."
            )

        expr = pd.read_csv(expression_csv)

        # Try merge on ENSID first
        if "ENSID" in expr.columns:
            expr["ENSID"] = (
                expr["ENSID"].astype(str).str.replace(r"\.\d+$", "", regex=True)
            )
            df = df.merge(
                expr[["ENSID", expression_column]].rename(
                    columns={expression_column: "Expression"}
                ),
                on="ENSID",
                how="left",
            )
        else:
            df["Expression"] = np.nan

        # Fallback: fill missing expression via gene symbol
        if df["Expression"].isna().any():
            symbol_col = None
            for candidate in ("symbol", "Gene", "gene_name", "gene_symbol"):
                if candidate in expr.columns:
                    symbol_col = candidate
                    break
            if symbol_col is not None:
                symbol_map = (
                    expr.dropna(subset=[expression_column])
                    .drop_duplicates(subset=[symbol_col])
                    .set_index(symbol_col)[expression_column]
                )
                mask = df["Expression"].isna()
                df.loc[mask, "Expression"] = df.loc[mask, "symbol"].map(symbol_map)

        # Override TSS if the expression CSV has a TSS column
        if tss_column in expr.columns:
            if "ENSID" in expr.columns:
                tss_map = expr.drop_duplicates(subset=["ENSID"]).set_index("ENSID")
                if tss_column in tss_map.columns:
                    mapped = df["ENSID"].map(tss_map[tss_column])
                    valid = mapped.notna()
                    df.loc[valid, "tss"] = mapped[valid].astype(int)

    # Select final columns
    keep = ["chr", "start", "end", "symbol", "ENSID", "strand", "gene_type", "tss"]
    if "Expression" in df.columns:
        keep.append("Expression")
    df = df[keep].copy()

    return df
