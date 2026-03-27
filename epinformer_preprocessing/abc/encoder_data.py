"""
Generate sequence encoder training data (Step 4).

Produces 256bp sequences with activity labels for pre-training
the ``enhancer_predictor_256bp`` model.  For each candidate peak,
five bins of 256bp are extracted at offsets [-3, -2, -1, 0, +1]
relative to the summit.  Optional negative samples are added from
randomly chosen genomic positions far from any peak.
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
_OFFSETS = [-3, -2, -1, 0, 1]
_NEG_EXCLUSION_FLANK = 1000  # exclude regions within 1 kb of any peak


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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

    records = []
    for _ in range(n_draw):
        idx = rng.choices(range(len(chroms)), weights=weights, k=1)[0]
        chrom = chroms[idx]
        clen = chrom_sizes[chrom]
        # Need room for the widest offset bin: center - 3*256 - 128
        margin = abs(min(_OFFSETS)) * _BIN_SIZE + _HALF_BIN
        pos = rng.randint(margin, clen - margin - 1)
        records.append((chrom, pos - _HALF_BIN, pos + _HALF_BIN, pos))

    candidate_bed = pybedtools.BedTool(
        [(r[0], r[1], r[2]) for r in records]
    ).sort()

    # Subtract exclusion zones
    surviving = candidate_bed.intersect(exclusion, v=True)

    # Collect surviving intervals back into a list
    kept = []
    for feat in surviving:
        chrom = feat.chrom
        start = int(feat.start)
        end = int(feat.end)
        summit = (start + end) // 2
        kept.append((chrom, start, end, summit))
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

def _extract_chrom_sequences(fasta_path, chrom, chrom_rows, chrom_sizes, has_h3k27ac):
    """Extract 5-bin sequences for all peaks on one chromosome (thread worker)."""
    fasta = pyfaidx.Fasta(fasta_path)
    rows = []
    for _, row in tqdm(chrom_rows.iterrows(), total=len(chrom_rows),
                       desc=f"  {chrom}", leave=False):
        summit = int(row["summit"])
        name = row["name"]
        dhs_rpm = float(row.get("DHS.RPM", 0.0))
        h3k27ac_rpm = float(row.get("H3K27ac.RPM", 0.0)) if has_h3k27ac else 0.0
        activity = float(row.get("activity_base", dhs_rpm))

        for offset in _OFFSETS:
            center = summit + offset * _BIN_SIZE
            start = center - _HALF_BIN
            end = center + _HALF_BIN

            seq = _extract_sequence(fasta, chrom, start, end, chrom_sizes)
            if seq is None:
                continue

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
    enhancer_list_path,
    candidates_summits_bed,
    fasta_path,
    chrom_sizes_path,
    output_dir,
    logger,
    cell_type="K562",
    neg_fraction=0.05,
    blacklist=None,
    n_threads=1,
    accessibility_bam=None,
    h3k27ac_bam=None,
    max_peaks=100_000,
):
    """Generate 256bp sequences + activity labels for encoder pre-training.

    Both positive (peak) and negative (random) regions are scored using
    the same activity pipeline: BAM read counting → RPM → quantile
    normalization → activity_base.

    Parameters
    ----------
    enhancer_list_path : str
        Path to EnhancerList.txt from Step 2.
    candidates_summits_bed : str
        Path to candidates_with_summits.bed (columns: chr, start, end, name,
        summit).
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
    neg_fraction : float
        Fraction of negative samples relative to peak count (default 0.05).
    blacklist : str or None
        Optional path to a blacklist BED file.
    accessibility_bam : str
        Path to the DNase/ATAC BAM for read counting.
    h3k27ac_bam : str or None
        Path to the H3K27ac BAM for read counting.

    Returns
    -------
    str
        Path to the output CSV file.
    """
    # -----------------------------------------------------------------
    # 1. Load EnhancerList and summit positions, then merge
    # -----------------------------------------------------------------
    enh = pd.read_csv(enhancer_list_path, sep="\t")
    summits = pd.read_csv(
        candidates_summits_bed,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "name", "summit"],
    )

    # Use interval overlap to match summits to EnhancerList regions.
    # Summit coordinates are pre-merge (original MACS2 peaks ±250bp),
    # while EnhancerList has post-merge coordinates, so exact matching
    # loses ~half the data.  Instead, intersect summit midpoints with
    # EnhancerList intervals.
    summit_points = pybedtools.BedTool([
        (row["chr"], int(row["summit"]), int(row["summit"]) + 1, str(i))
        for i, row in summits.iterrows()
    ]).sort()
    enh_bed = pybedtools.BedTool([
        (row["chr"], int(row["start"]), int(row["end"]), str(i))
        for i, row in enh.iterrows()
    ]).sort()
    hits = summit_points.intersect(enh_bed, wa=True, wb=True)

    # Build merged DataFrame: each hit maps a summit to an EnhancerList row
    merge_rows = []
    for feat in hits:
        summit_pos = int(feat.fields[1])  # summit point
        enh_idx = int(feat.fields[7])     # EnhancerList row index
        merge_rows.append({"summit": summit_pos, "_enh_idx": enh_idx})

    if merge_rows:
        hits_df = pd.DataFrame(merge_rows).drop_duplicates(subset=["_enh_idx"])
        merged = enh.loc[hits_df["_enh_idx"].values].copy().reset_index(drop=True)
        merged["summit"] = hits_df["summit"].values
    else:
        # Fallback to exact match if interval overlap finds nothing
        merged = enh.merge(summits[["chr", "start", "end", "summit"]],
                           on=["chr", "start", "end"], how="inner")
    pybedtools.cleanup()

    has_h3k27ac = "H3K27ac.RPM" in merged.columns and merged["H3K27ac.RPM"].sum() > 0

    logger.info(f"Loaded {len(merged)} candidates with summits (H3K27ac={'yes' if has_h3k27ac else 'no'})")

    # -----------------------------------------------------------------
    # 2. Load chromosome sizes and FASTA
    # -----------------------------------------------------------------
    chrom_sizes = _load_chrom_sizes(chrom_sizes_path)

    # -----------------------------------------------------------------
    # 3. Extract 5 bins per candidate peak (parallel by chromosome)
    # -----------------------------------------------------------------
    grouped = {chrom: grp for chrom, grp in merged.groupby("chr")}

    if n_threads <= 1 or len(grouped) <= 1:
        rows = []
        for chrom, grp in tqdm(grouped.items(), desc="  Extracting sequences", leave=False):
            rows.extend(_extract_chrom_sequences(fasta_path, chrom, grp, chrom_sizes, has_h3k27ac))
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=min(n_threads, len(grouped))) as pool:
            futures = {
                chrom: pool.submit(_extract_chrom_sequences, fasta_path, chrom, grp, chrom_sizes, has_h3k27ac)
                for chrom, grp in grouped.items()
            }
            for chrom, future in tqdm(futures.items(), total=len(futures),
                                      desc="  Extracting sequences", leave=False):
                rows.extend(future.result())

    logger.info(f"Extracted {len(rows)} sequence bins from {len(merged)} peaks")

    # -----------------------------------------------------------------
    # 4. Generate negative samples
    # -----------------------------------------------------------------
    n_neg = max(1, int(len(merged) * neg_fraction))

    # Build a BedTool from peaks for exclusion
    peak_records = [
        (str(r["chr"]), int(r["start"]), int(r["end"]))
        for _, r in merged.iterrows()
    ]
    peak_bed = pybedtools.BedTool(peak_records).sort()

    neg_df = _generate_negative_regions(
        peak_bed,
        chrom_sizes,
        n_neg,
        blacklist=blacklist,
    )

    # -----------------------------------------------------------------
    # 4b. Compute activity for ALL regions (positive + negative) via BAM
    # -----------------------------------------------------------------
    from .utils import count_reads_in_regions

    # Combine positive and negative regions into one DataFrame
    pos_regions = merged[["chr", "start", "end", "name", "summit"]].copy()
    n_pos = len(pos_regions)
    all_regions = pd.concat([pos_regions, neg_df], ignore_index=True)

    # Count DNase/ATAC reads
    logger.info("Counting accessibility reads for all regions ...")
    all_regions = count_reads_in_regions(accessibility_bam, all_regions, n_threads=n_threads)
    all_regions.rename(columns={
        "readCount": "DHS.readCount", "RPM": "DHS.RPM", "RPKM": "DHS.RPKM",
    }, inplace=True)

    # Count H3K27ac reads (if BAM provided)
    _has_h3k27ac_bam = h3k27ac_bam is not None
    if _has_h3k27ac_bam:
        logger.info("Counting H3K27ac reads for all regions ...")
        h3k_counts = count_reads_in_regions(h3k27ac_bam, all_regions, n_threads=n_threads)
        all_regions["H3K27ac.RPM"] = h3k_counts["RPM"]

    # Compute activity_base from raw RPM
    # With H3K27ac: geometric mean of both signals
    # Without: DNase RPM alone
    if _has_h3k27ac_bam:
        all_regions["activity_base"] = np.sqrt(
            all_regions["H3K27ac.RPM"] * all_regions["DHS.RPM"]
        )
    else:
        all_regions["activity_base"] = all_regions["DHS.RPM"]

    # Split back into positive and negative
    merged = all_regions.iloc[:n_pos].copy()
    neg_df = all_regions.iloc[n_pos:].copy()

    # Keep only top-N peaks by activity
    if max_peaks and len(merged) > max_peaks:
        merged = merged.nlargest(max_peaks, "activity_base").reset_index(drop=True)
        logger.info(f"Kept top {max_peaks} peaks by activity (dropped {n_pos - max_peaks})")

    # Update has_h3k27ac flag based on measured data
    has_h3k27ac = "H3K27ac.RPM" in merged.columns and merged["H3K27ac.RPM"].sum() > 0

    logger.info(
        f"Activity computed from BAMs — "
        f"positive mean: {merged['activity_base'].mean():.4f}, "
        f"negative mean: {neg_df['activity_base'].mean():.4f}"
    )

    neg_grouped = {chrom: grp for chrom, grp in neg_df.groupby("chr")}
    if n_threads <= 1 or len(neg_grouped) <= 1:
        neg_rows = []
        for chrom, grp in tqdm(neg_grouped.items(), desc="  Negative sequences", leave=False):
            neg_rows.extend(_extract_chrom_sequences(fasta_path, chrom, grp, chrom_sizes, has_h3k27ac))
    else:
        neg_rows = []
        with ThreadPoolExecutor(max_workers=min(n_threads, len(neg_grouped))) as pool:
            futures = {
                chrom: pool.submit(_extract_chrom_sequences, fasta_path, chrom, grp, chrom_sizes, has_h3k27ac)
                for chrom, grp in neg_grouped.items()
            }
            for chrom, future in tqdm(futures.items(), total=len(futures),
                                      desc="  Negative sequences", leave=False):
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
        f"{cell_type}_peak_5bins_around_summit_activity_sequence.csv",
    )
    out_df.to_csv(out_path, index=False)

    logger.info(f"Result: {len(out_df)} sequence-activity pairs")
    logger.info(f"Including {len(neg_rows)} negative samples")

    return out_path
