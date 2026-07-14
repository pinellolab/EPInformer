"""
Generate sequence encoder training data (Step 4).

Produces 256bp sequences with activity labels for pre-training
the ``enhancer_predictor_256bp`` model.  Peaks/summits come from the
ENCODE H3K27ac narrowPeak (``encoder_peaks_file`` in samples.tsv; the
10th column is the summit offset) — this is preferred over the DNase
MACS2 candidates so the windows match the H3K27ac activity being
predicted.  For each summit, five 256bp windows are extracted at
offsets [-2, -1, 0, 1, 2] (summit + two flanks each side) with a 192bp
stride, i.e. adjacent windows overlap by 64bp.  This matches the original
BSCC_GPU 5-bin recipe that reaches ~0.70 pooled-OOF Pearson.  Activity
(DNase RPM, or √(H3K27ac·DNase) when both BAMs are provided) is computed
**per 256bp bin**, not over the full peak interval.  Optional negative
samples are added from randomly chosen genomic positions far from any peak.
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyfaidx
import pybedtools
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BIN_SIZE = 256
_HALF_BIN = _BIN_SIZE // 2  # 128
# Bin geometry is env-overridable so different recipes (2/3/5-bin) can be generated
# without editing this file.
#   Defaults = BSCC_GPU 5-bin recipe: offsets [-2..2], 192bp stride (64bp overlap).
#   Published 2-bin recipe: ENCODER_OFFSETS="0,1" ENCODER_OVERLAP=100 (156bp stride).
_OVERLAP = int(os.environ.get("ENCODER_OVERLAP", "64"))
_STRIDE = _BIN_SIZE - _OVERLAP
_OFFSETS = [int(x) for x in os.environ.get("ENCODER_OFFSETS", "-2,-1,0,1,2").split(",")]
# Invariant: there is ALWAYS a window centered exactly on the summit (offset 0) —
# it is the single most informative bin (the peak call's actual summit). Guarantee
# it even if a custom ENCODER_OFFSETS omits it, and drop any duplicate offsets.
_OFFSETS = sorted(set(_OFFSETS) | {0})
_NEG_EXCLUSION_FLANK = 1000  # exclude regions within 1 kb of any peak

_NARROWPEAK_COLUMNS = [
    "chr", "start", "end", "name", "score", "strand",
    "signalValue", "pValue", "qValue", "peak",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_reads_mean_reps(bam_paths, regions, n_threads, logger=None, label=""):
    """Count reads in *regions* across one or more replicate BAMs and return
    the per-rep-averaged RPM/RPKM/readCount.

    If *bam_paths* is a single string, this collapses to the original
    single-rep behavior.  If it is a list/tuple, each rep's RPM is computed
    independently and the per-region mean across reps is returned (matching
    the legacy BSCC_GPU pipeline).
    """
    from .utils import count_reads_in_regions

    if isinstance(bam_paths, str):
        bam_paths = [bam_paths]
    bam_paths = [b for b in bam_paths if b]
    if not bam_paths:
        raise ValueError("bam_paths is empty")

    if logger is not None and len(bam_paths) > 1:
        logger.info(f"  {label}: averaging across {len(bam_paths)} biological replicates")

    rep_dfs = []
    for i, bam in enumerate(bam_paths):
        if logger is not None and len(bam_paths) > 1:
            logger.info(f"    rep{i}: {bam}")
        out = count_reads_in_regions(bam, regions[["chr", "start", "end"]], n_threads=n_threads)
        rep_dfs.append(out[["readCount", "RPM", "RPKM"]].reset_index(drop=True))

    merged = regions.reset_index(drop=True).copy()
    if len(rep_dfs) == 1:
        merged[["readCount", "RPM", "RPKM"]] = rep_dfs[0]
        return merged

    stack = np.stack([d.values for d in rep_dfs], axis=0)  # (R, N, 3)
    merged[["readCount", "RPM", "RPKM"]] = stack.mean(axis=0)
    return merged


def _extract_sequence(fasta, chrom, start, end, chrom_sizes):
    """Fetch an upper-case sequence from *fasta*, or None if out of bounds."""
    if start < 0 or end > chrom_sizes.get(chrom, 0):
        return None
    try:
        return fasta[chrom][start:end].seq.upper()
    except (KeyError, ValueError):
        return None


def _load_chrom_sizes(path):
    """Return a dict mapping chromosome name → length from a TSV file."""
    df = pd.read_csv(path, sep="\t", header=None, names=["chrom", "size"])
    return dict(zip(df["chrom"], df["size"]))


def _normalize_macs_peak_name(name):
    """MACS2 ``callpeak -n peaks`` emits names like ``peaks_peak_123``; collapse to ``peaks_123``."""
    s = str(name)
    if s.startswith("peaks_peak_"):
        return "peaks_" + s[len("peaks_peak_") :]
    return s


def _build_bin_intervals(df, chrom_sizes, id_col="peak_idx", summit_col="summit"):
    """256bp windows matching ``_extract_chrom_sequences`` (one row per valid bin)."""
    records = []
    for _, row in df.iterrows():
        chrom = row["chr"]
        clen = chrom_sizes.get(chrom, 0)
        summit = int(row[summit_col])
        pid = int(row[id_col])
        name = row["name"]
        for offset in _OFFSETS:
            center = summit + offset * _STRIDE
            start = center - _HALF_BIN
            end = center + _HALF_BIN
            if start < 0 or end > clen:
                continue
            records.append({
                id_col: pid,
                "name": name,
                "chr": chrom,
                "summit": summit,
                "offset": offset,
                "start": int(start),
                "end": int(end),
            })
    return pd.DataFrame(records)


def _generate_negative_regions(
    peak_bed,
    chrom_sizes,
    n_neg,
    blacklist=None,
    seed=42,
):
    """Sample random genomic intervals that do not overlap peaks (±1 kb).

    Returns a DataFrame with columns: chr, start, end, name, summit.
    """
    rng = random.Random(seed)

    # Build an exclusion BedTool: peaks flanked by 1 kb
    # pybedtools expects {chrom: (0, length)} tuples for the g= parameter
    chrom_sizes_bed = {c: (0, s) for c, s in chrom_sizes.items()}
    exclusion = peak_bed.slop(
        b=_NEG_EXCLUSION_FLANK,
        g=chrom_sizes_bed,
    ).sort().merge()

    if blacklist is not None:
        bl_bed = pybedtools.BedTool(blacklist)
        exclusion = exclusion.cat(bl_bed, postmerge=True)

    # Weighted chromosome sampling by length
    chroms = sorted(chrom_sizes.keys())
    # Keep only standard chromosomes (chr1..chr22, chrX, chrY)
    valid = {f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]}
    chroms = [c for c in chroms if c in valid]
    lengths = np.array([chrom_sizes[c] for c in chroms], dtype=float)
    weights = lengths / lengths.sum()

    # Over-sample to account for exclusion filtering
    oversample_factor = 3
    n_draw = n_neg * oversample_factor

    # Full reach of the 5-bin window around a center: the widest offset bin spans
    # out to center +/- (|max offset|*stride + half bin). We test this ENTIRE
    # footprint against the exclusion zone (not just the central 256bp bin), so a
    # negative can't have any of its offset bins fall inside a peak neighborhood.
    margin = abs(max(_OFFSETS, key=abs)) * _STRIDE + _HALF_BIN

    records = []
    for _ in range(n_draw):
        idx = rng.choices(range(len(chroms)), weights=weights, k=1)[0]
        chrom = chroms[idx]
        clen = chrom_sizes[chrom]
        pos = rng.randint(margin, clen - margin - 1)
        records.append((chrom, pos))

    candidate_bed = pybedtools.BedTool(
        [(chrom, pos - margin, pos + margin) for chrom, pos in records]
    ).sort()

    # Keep only centers whose full 5-bin footprint avoids the exclusion zone
    surviving = candidate_bed.intersect(exclusion, v=True)

    # Collect surviving intervals; store the CENTRAL bin (consistent with positives),
    # summit = the sampled center (= midpoint of the symmetric footprint).
    kept = []
    for feat in surviving:
        summit = (int(feat.start) + int(feat.end)) // 2
        kept.append((feat.chrom, summit - _HALF_BIN, summit + _HALF_BIN, summit))
        if len(kept) >= n_neg:
            break

    rows = []
    for i, (chrom, start, end, summit) in enumerate(kept, 1):
        rows.append({
            "chr": chrom,
            "start": start,
            "end": end,
            "name": f"Neg_{i}",
            "summit": summit,
        })

    pybedtools.cleanup()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_chrom_sequences(
    fasta_path,
    chrom,
    chrom_rows,
    chrom_sizes,
    has_h3k27ac,
    bin_lookup,
    id_col,
):
    """Extract 5-bin sequences; RPM and Activity come from *bin_lookup* (per 256bp bin)."""
    fasta = pyfaidx.Fasta(fasta_path)
    rows = []
    for _, row in tqdm(chrom_rows.iterrows(), total=len(chrom_rows),
                       desc=f"  {chrom}", leave=False, ncols=80):
        summit = int(row["summit"])
        name = row["name"]
        pid = int(row[id_col])

        for offset in _OFFSETS:
            key = (pid, offset)
            if key not in bin_lookup.index:
                continue

            center = summit + offset * _STRIDE
            start = center - _HALF_BIN
            end = center + _HALF_BIN

            seq = _extract_sequence(fasta, chrom, start, end, chrom_sizes)
            if seq is None:
                continue

            stats = bin_lookup.loc[key]
            dhs_rpm = float(stats["DHS.RPM"])
            h3k27ac_rpm = float(stats["H3K27ac.RPM"]) if has_h3k27ac else 0.0
            activity = float(stats["activity_base"])

            rows.append({
                "Name": name,
                "Chromosome": chrom,
                "Start": start,
                "End": end,
                "Summit": summit,
                "Offset_to_summit": offset,
                "DNase_RPM": dhs_rpm,
                "H3K27ac_RPM": h3k27ac_rpm,
                "Sequence": seq,
                "Activity": activity,
            })
    fasta.close()
    return rows


def generate_encoder_data(
    narrowpeak_path,
    fasta_path,
    chrom_sizes_path,
    output_dir,
    logger,
    cell_type="K562",
    max_peaks=100_000,
    neg_fraction=0.05,
    blacklist=None,
    n_threads=1,
    accessibility_bam=None,
    h3k27ac_bam=None,
    accessibility_bams=None,
    h3k27ac_bams=None,
):
    """Generate 256bp sequences + activity labels for encoder pre-training.

    Reads the raw MACS2 narrowPeak file, selects the top *max_peaks* summits
    by signalValue, counts BAM reads **in each 256bp bin** for activity labels,
    extracts 5 bins around each summit, and adds negative samples.

    Parameters
    ----------
    narrowpeak_path : str
        Path to a MACS2 narrowPeak file (10-column, tab-separated).
    fasta_path : str
        Path to the hg38.fa reference genome FASTA.
    chrom_sizes_path : str
        Path to a two-column TSV of chromosome sizes.
    output_dir : str
        Directory for the output CSV.
    logger : StepLogger
        Logger instance for progress messages.
    cell_type : str
        Cell-type label used in the output filename (default ``"K562"``).
    max_peaks : int
        Number of top peaks to select by signalValue (default 100,000).
    neg_fraction : float
        Fraction of negative samples relative to peak count (default 0.05).
    blacklist : str or None
        Optional path to a blacklist BED file.
    n_threads : int
        Number of threads for parallel sequence extraction.
    accessibility_bam : str
        Path to the DNase/ATAC BAM for read counting (single rep).
    h3k27ac_bam : str or None
        Path to the H3K27ac BAM for read counting (single rep).
    accessibility_bams : list[str] or None
        Multi-rep override for ``accessibility_bam``.  When provided, RPM is
        averaged across these biological-replicate BAMs (legacy-style).
    h3k27ac_bams : list[str] or None
        Multi-rep override for ``h3k27ac_bam``.  When provided, RPM is
        averaged across these biological-replicate BAMs (legacy-style).

    Returns
    -------
    str
        Path to the output CSV file.
    """
    # -----------------------------------------------------------------
    # 1. Load narrowPeak, select top peaks by signalValue
    # -----------------------------------------------------------------
    peaks = pd.read_csv(
        narrowpeak_path,
        sep="\t",
        header=None,
        names=_NARROWPEAK_COLUMNS,
        comment="#",
    )
    n_total = len(peaks)
    n_select = min(max_peaks, n_total)
    peaks = peaks.nlargest(n_select, "signalValue").reset_index(drop=True)
    peaks["summit"] = peaks["start"] + peaks["peak"]
    peaks["name"] = peaks["name"].map(_normalize_macs_peak_name)

    logger.info(f"Selected top {n_select} summits by signalValue (from {n_total} total peaks)")

    # -----------------------------------------------------------------
    # 2. Count BAM reads per 256bp bin (matches sequence windows)
    # -----------------------------------------------------------------
    chrom_sizes = _load_chrom_sizes(chrom_sizes_path)

    # Invariant: every emitted peak must have a window AT the summit (offset 0) —
    # its most informative bin. Drop peaks whose summit-centered 256bp window falls
    # off the chromosome (near an edge, or on a contig absent from chrom_sizes), so
    # we never emit a peak represented by surviving flanks with no summit window.
    _clen = peaks["chr"].map(chrom_sizes).fillna(0).astype(np.int64)
    _summit_ok = (peaks["summit"] - _HALF_BIN >= 0) & (peaks["summit"] + _HALF_BIN <= _clen)
    n_drop = int((~_summit_ok).sum())
    if n_drop:
        logger.info(f"Dropped {n_drop} peak(s) whose summit window falls off the chromosome edge")
    peaks = peaks[_summit_ok].reset_index(drop=True)

    regions = peaks[["chr", "start", "end", "name", "summit"]].copy()
    n_pos = len(regions)
    regions["peak_idx"] = np.arange(n_pos, dtype=np.int64)
    merged = regions

    n_neg = max(1, int(n_pos * neg_fraction))
    peak_bed = pybedtools.BedTool([
        (str(r["chr"]), int(r["start"]), int(r["end"]))
        for _, r in regions.iterrows()
    ]).sort()
    neg_df = _generate_negative_regions(peak_bed, chrom_sizes, n_neg, blacklist=blacklist)
    neg_df["neg_idx"] = np.arange(len(neg_df), dtype=np.int64)

    bin_pos = _build_bin_intervals(merged, chrom_sizes, id_col="peak_idx")
    bin_neg = _build_bin_intervals(neg_df, chrom_sizes, id_col="neg_idx")
    n_pos_bins = len(bin_pos)
    bin_all = pd.concat([bin_pos, bin_neg], ignore_index=True)

    dnase_paths = accessibility_bams if accessibility_bams else accessibility_bam
    logger.info(f"Counting accessibility reads in {len(bin_all)} 256bp bins ...")
    bin_all = _count_reads_mean_reps(
        dnase_paths, bin_all, n_threads=n_threads, logger=logger, label="DNase",
    )
    bin_all.rename(columns={
        "readCount": "DHS.readCount", "RPM": "DHS.RPM", "RPKM": "DHS.RPKM",
    }, inplace=True)

    h3k_paths = h3k27ac_bams if h3k27ac_bams else h3k27ac_bam
    has_h3k27ac = bool(h3k_paths)
    if has_h3k27ac:
        logger.info("Counting H3K27ac reads per 256bp bin ...")
        h3k = _count_reads_mean_reps(
            h3k_paths, bin_all[["chr", "start", "end"]], n_threads=n_threads,
            logger=logger, label="H3K27ac",
        )
        bin_all["H3K27ac.RPM"] = h3k["RPM"].values
    else:
        bin_all["H3K27ac.RPM"] = 0.0

    if has_h3k27ac:
        bin_all["activity_base"] = np.sqrt(
            bin_all["H3K27ac.RPM"] * bin_all["DHS.RPM"]
        )
    else:
        bin_all["activity_base"] = bin_all["DHS.RPM"]

    pos_bins_df = bin_all.iloc[:n_pos_bins].copy()
    neg_bins_df = bin_all.iloc[n_pos_bins:].copy()

    pos_lookup = pos_bins_df.set_index(["peak_idx", "offset"])[
        ["DHS.RPM", "H3K27ac.RPM", "activity_base"]
    ]
    neg_lookup = neg_bins_df.set_index(["neg_idx", "offset"])[
        ["DHS.RPM", "H3K27ac.RPM", "activity_base"]
    ]

    has_h3k27ac_col = has_h3k27ac and bin_all["H3K27ac.RPM"].sum() > 0

    logger.info(
        f"Per-bin activity from BAMs — "
        f"positive bins mean: {pos_bins_df['activity_base'].mean():.4f}, "
        f"negative bins mean: {neg_bins_df['activity_base'].mean():.4f}"
    )

    # -----------------------------------------------------------------
    # 3. Extract 5 bins per summit (parallel by chromosome)
    # -----------------------------------------------------------------
    grouped = {chrom: grp for chrom, grp in merged.groupby("chr")}

    if n_threads <= 1 or len(grouped) <= 1:
        rows = []
        for chrom, grp in tqdm(grouped.items(), desc="  Extracting sequences", leave=False, ncols=80):
            rows.extend(_extract_chrom_sequences(
                fasta_path, chrom, grp, chrom_sizes, has_h3k27ac_col, pos_lookup, "peak_idx",
            ))
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=min(n_threads, len(grouped))) as pool:
            futures = {
                chrom: pool.submit(
                    _extract_chrom_sequences,
                    fasta_path, chrom, grp, chrom_sizes, has_h3k27ac_col, pos_lookup, "peak_idx",
                )
                for chrom, grp in grouped.items()
            }
            for chrom, future in tqdm(futures.items(), total=len(futures),
                                      desc="  Extracting sequences", leave=False, ncols=80):
                rows.extend(future.result())

    logger.info(f"Extracted {len(rows)} sequence bins from {n_pos} peaks")

    # -----------------------------------------------------------------
    # 4. Extract sequences for negative samples
    # -----------------------------------------------------------------
    neg_grouped = {chrom: grp for chrom, grp in neg_df.groupby("chr")}
    if n_threads <= 1 or len(neg_grouped) <= 1:
        neg_rows = []
        for chrom, grp in tqdm(neg_grouped.items(), desc="  Negative sequences", leave=False, ncols=80):
            neg_rows.extend(_extract_chrom_sequences(
                fasta_path, chrom, grp, chrom_sizes, has_h3k27ac_col, neg_lookup, "neg_idx",
            ))
    else:
        neg_rows = []
        with ThreadPoolExecutor(max_workers=min(n_threads, len(neg_grouped))) as pool:
            futures = {
                chrom: pool.submit(
                    _extract_chrom_sequences,
                    fasta_path, chrom, grp, chrom_sizes, has_h3k27ac_col, neg_lookup, "neg_idx",
                )
                for chrom, grp in neg_grouped.items()
            }
            for chrom, future in tqdm(futures.items(), total=len(futures),
                                      desc="  Negative sequences", leave=False, ncols=80):
                neg_rows.extend(future.result())

    logger.info(f"Generated {len(neg_rows)} negative sequence bins from {len(neg_df)} regions")

    # -----------------------------------------------------------------
    # 5. Combine and write output CSV
    # -----------------------------------------------------------------
    all_rows = rows + neg_rows
    out_df = pd.DataFrame(all_rows, columns=[
        "Name", "Chromosome", "Start", "End", "Summit",
        "Offset_to_summit", "DNase_RPM", "H3K27ac_RPM",
        "Sequence", "Activity",
    ])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"{cell_type}_peak_{len(_OFFSETS)}bins_around_summit_activity_sequence.csv",
    )
    out_df.to_csv(out_path, index=False)

    logger.info(f"Result: {len(out_df)} sequence-activity pairs")
    logger.info(f"Including {len(neg_rows)} negative samples")

    return out_path
