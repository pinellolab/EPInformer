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

import numpy as np
import pandas as pd
import pyfaidx
import pybedtools


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
    exclusion = peak_bed.slop(
        b=_NEG_EXCLUSION_FLANK,
        g=chrom_sizes,
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
):
    """Generate 256bp sequences + activity labels for encoder pre-training.

    Parameters
    ----------
    enhancer_list_path : str
        Path to EnhancerList.txt from Step 2 (tab-separated, columns include
        chr, start, end, name, DHS.RPM, H3K27ac.RPM, activity_base).
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

    # Merge on chr/start/end to get summit + activity for each candidate
    merged = enh.merge(summits[["chr", "start", "end", "summit"]], on=["chr", "start", "end"], how="inner")

    has_h3k27ac = "H3K27ac.RPM" in merged.columns and merged["H3K27ac.RPM"].sum() > 0

    logger.info(f"Loaded {len(merged)} candidates with summits (H3K27ac={'yes' if has_h3k27ac else 'no'})")

    # -----------------------------------------------------------------
    # 2. Load chromosome sizes and FASTA
    # -----------------------------------------------------------------
    chrom_sizes = _load_chrom_sizes(chrom_sizes_path)
    fasta = pyfaidx.Fasta(fasta_path)

    # -----------------------------------------------------------------
    # 3. Extract 5 bins per candidate peak
    # -----------------------------------------------------------------
    rows = []
    for _, row in merged.iterrows():
        chrom = str(row["chr"])
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

    # Convert chrom_sizes dict to a temp file for pybedtools
    cs_path_for_bedtools = chrom_sizes_path

    neg_df = _generate_negative_regions(
        peak_bed,
        cs_path_for_bedtools,
        n_neg,
        blacklist=blacklist,
    )

    neg_rows = []
    for _, row in neg_df.iterrows():
        chrom = str(row["chr"])
        summit = int(row["summit"])
        name = row["name"]

        for offset in _OFFSETS:
            center = summit + offset * _BIN_SIZE
            start = center - _HALF_BIN
            end = center + _HALF_BIN

            seq = _extract_sequence(fasta, chrom, start, end, chrom_sizes)
            if seq is None:
                continue

            neg_rows.append({
                "Name": name,
                "Chromosome": chrom,
                "Start": start,
                "End": end,
                "Summit": summit,
                "Offset_to_summit": offset,
                "DNase_RPM": 0.0,
                "H3K27ac_RPM": 0.0,
                "Sequence": seq,
                "Activity": 0.0,
            })

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
