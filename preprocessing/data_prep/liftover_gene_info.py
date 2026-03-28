#!/usr/bin/env python3
"""
LiftOver Ensembl v65 gene_info (hg19) to hg38 CollapsedGeneBounds BED format.

Converts the Roadmap ``Ensembl_v65.Gencode_v10.ENSG.gene_info`` file into a
CollapsedGeneBounds-compatible BED that can be used as the ABC pipeline gene
reference, ensuring perfect overlap with Roadmap expression data.

The liftOver binary and chain file are auto-downloaded if not found.

Examples::

    # Auto-downloads liftOver + chain file to data/reference/
    python scripts/liftover_gene_info.py \\
        --gene-info data/roadmap_expression/.cache/Ensembl_v65.Gencode_v10.ENSG.gene_info \\
        --output data/reference/hg38/CollapsedGeneBounds.from_gene_info.hg38.bed

    # With explicit paths
    python scripts/liftover_gene_info.py \\
        --gene-info data/roadmap_expression/.cache/Ensembl_v65.Gencode_v10.ENSG.gene_info \\
        --chain data/reference/hg19ToHg38.over.chain.gz \\
        --output data/reference/hg38/CollapsedGeneBounds.from_gene_info.hg38.bed
"""

from __future__ import annotations

import argparse
import gzip
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request

_CHAIN_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"
_LIFTOVER_URLS = {
    ("Darwin", "x86_64"): "https://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/liftOver",
    ("Darwin", "arm64"): "https://hgdownload.soe.ucsc.edu/admin/exe/macOSX.arm64/liftOver",
    ("Linux", "x86_64"): "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver",
}


def _download(url: str, dest: str) -> str:
    """Download a URL to dest if not already present (uses curl to avoid SSL issues)."""
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {url} ...")
    result = subprocess.run(
        ["curl", "-fSL", "-o", dest, url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")
    return dest


def _ensure_liftover(cache_dir: str, explicit_bin: str | None) -> str:
    """Return path to liftOver binary, downloading if needed."""
    if explicit_bin and explicit_bin != "liftOver":
        return explicit_bin
    # Check if already on PATH
    if shutil.which("liftOver"):
        return "liftOver"
    # Download
    key = (platform.system(), platform.machine())
    url = _LIFTOVER_URLS.get(key)
    if url is None:
        raise RuntimeError(
            f"No liftOver binary available for {key}. "
            f"Download manually from https://hgdownload.soe.ucsc.edu/admin/exe/"
        )
    dest = os.path.join(cache_dir, "liftOver")
    _download(url, dest)
    os.chmod(dest, os.stat(dest).st_mode | stat.S_IEXEC)
    return dest


def _ensure_chain(cache_dir: str, explicit_chain: str | None) -> str:
    """Return path to chain file, downloading + decompressing if needed."""
    if explicit_chain and os.path.exists(explicit_chain):
        return explicit_chain
    # Check for decompressed version
    chain_path = os.path.join(cache_dir, "hg19ToHg38.over.chain")
    if os.path.exists(chain_path):
        return chain_path
    # Download and decompress
    gz_path = os.path.join(cache_dir, "hg19ToHg38.over.chain.gz")
    _download(_CHAIN_URL, gz_path)
    print(f"  Decompressing {gz_path} ...")
    with gzip.open(gz_path, "rb") as f_in, open(chain_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return chain_path


def _load_gene_info(path: str) -> list[dict]:
    """Parse the Ensembl v65 gene_info file (handles 8-10 field lines)."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            rows.append({
                "ENSID": parts[0],
                "chrom": parts[1],
                "start": int(parts[2]),
                "end": int(parts[3]),
                "strand_num": int(parts[4]),
                "biotype": parts[5],
                "gene_name": parts[6] if len(parts) > 6 else "",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiftOver gene_info (hg19) to hg38 CollapsedGeneBounds BED"
    )
    parser.add_argument(
        "--gene-info", required=True,
        help="Path to Ensembl_v65.Gencode_v10.ENSG.gene_info",
    )
    parser.add_argument(
        "--chain", default=None,
        help="Path to hg19ToHg38.over.chain(.gz). Auto-downloaded if omitted.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output BED file path (CollapsedGeneBounds format)",
    )
    parser.add_argument(
        "--liftover-bin", default="liftOver",
        help="Path to UCSC liftOver binary (default: liftOver on PATH)",
    )
    parser.add_argument(
        "--min-match", type=float, default=0.5,
        help="Minimum ratio of bases that must remap (default: 0.5)",
    )
    parser.add_argument(
        "--compare", default=None,
        help="Path to original CollapsedGeneBounds.hg38.bed for discrepancy report.",
    )
    parser.add_argument(
        "--biotypes", nargs="+", default=["protein_coding", "lincRNA"],
        help="Biotypes to keep (default: protein_coding lincRNA). Use 'all' to keep everything.",
    )
    args = parser.parse_args()

    # ---- 0. Ensure liftOver binary and chain file ----
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), ".liftover_cache")
    liftover_bin = _ensure_liftover(cache_dir, args.liftover_bin)
    chain_path = _ensure_chain(cache_dir, args.chain)
    print(f"liftOver binary: {liftover_bin}")
    print(f"Chain file:      {chain_path}")

    # ---- 1. Parse gene_info ----
    print(f"Loading gene_info: {args.gene_info}")
    genes = _load_gene_info(args.gene_info)
    print(f"  Loaded {len(genes)} genes")

    if "all" not in args.biotypes:
        keep = set(args.biotypes)
        genes = [g for g in genes if g["biotype"] in keep]
        print(f"  Filtered to {len(genes)} genes (biotypes: {', '.join(sorted(keep))})")

    # Build lookup by ENSID for later rejoin
    gene_lookup = {g["ENSID"]: g for g in genes}

    # ---- 2. Write hg19 BED for liftOver ----
    # BED format: chr  start  end  name  score  strand
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".bed", delete=False, prefix="gene_info_hg19_"
    ) as tmp_in:
        tmp_in_path = tmp_in.name
        for g in genes:
            chrom = g["chrom"]
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            strand = "+" if g["strand_num"] == 1 else "-"
            tmp_in.write(
                f"{chrom}\t{g['start']}\t{g['end']}\t{g['ENSID']}\t0\t{strand}\n"
            )

    # ---- 3. Run liftOver ----
    tmp_out_path = tmp_in_path.replace("hg19_", "hg38_")
    tmp_unmapped_path = tmp_in_path + ".unmapped"

    cmd = [
        liftover_bin,
        tmp_in_path,
        chain_path,
        tmp_out_path,
        tmp_unmapped_path,
        "-minMatch=" + str(args.min_match),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"liftOver stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"liftOver failed with return code {result.returncode}")

    # Count unmapped
    n_unmapped = 0
    if os.path.exists(tmp_unmapped_path):
        with open(tmp_unmapped_path) as f:
            n_unmapped = sum(1 for line in f if not line.startswith("#"))

    # ---- 4. Read lifted BED and convert to CollapsedGeneBounds format ----
    lifted = []
    with open(tmp_out_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            ensid = parts[3]
            strand = parts[5]

            g = gene_lookup.get(ensid)
            if g is None:
                continue

            lifted.append({
                "chr": chrom,
                "start": start,
                "end": end,
                "symbol": g["gene_name"],
                "score": 0,
                "strand": strand,
                "ENSID": ensid,
                "gene_type": g["biotype"],
            })

    print(f"  Lifted: {len(lifted)} genes")
    print(f"  Unmapped: {n_unmapped} genes")

    # Deduplicate by ENSID (keep first occurrence)
    seen = set()
    deduped = []
    for g in lifted:
        if g["ENSID"] not in seen:
            seen.add(g["ENSID"])
            deduped.append(g)
    if len(deduped) < len(lifted):
        print(f"  Deduplicated: {len(lifted)} → {len(deduped)} unique ENSIDs")
    lifted = deduped

    # ---- 5. Write CollapsedGeneBounds BED ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Sort by chromosome and start position
    lifted.sort(key=lambda x: (x["chr"], x["start"]))

    with open(args.output, "w") as f:
        f.write("#chr\tstart\tend\tname\tscore\tstrand\tEnsembl_ID\tgene_type\n")
        for g in lifted:
            f.write(
                f"{g['chr']}\t{g['start']}\t{g['end']}\t{g['symbol']}\t"
                f"{g['score']}\t{g['strand']}\t{g['ENSID']}\t{g['gene_type']}\n"
            )

    print(f"Output: {args.output} ({len(lifted)} genes)")

    # ---- 6. Compare with original CollapsedGeneBounds (optional) ----
    if args.compare:
        print(f"\n{'=' * 60}")
        print(f"Comparing with: {args.compare}")
        print(f"{'=' * 60}")

        abc_genes = {}  # ENSID → (chr, start, end, symbol)
        with open(args.compare) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                ensid = parts[6]
                abc_genes[ensid] = {
                    "chr": parts[0], "start": int(parts[1]),
                    "end": int(parts[2]), "symbol": parts[3],
                }

        lifted_ensids = {g["ENSID"] for g in lifted}
        abc_ensids = set(abc_genes.keys())

        overlap = lifted_ensids & abc_ensids
        lifted_only = lifted_ensids - abc_ensids
        abc_only = abc_ensids - lifted_ensids

        print(f"  Lifted (gene_info→hg38):   {len(lifted_ensids)}")
        print(f"  ABC (CollapsedGeneBounds): {len(abc_ensids)}")
        print(f"  Overlap:                   {len(overlap)}")
        print(f"  In lifted only:            {len(lifted_only)}")
        print(f"  In ABC only:               {len(abc_only)}")

        # Check coordinate differences for overlapping genes
        lifted_lookup = {g["ENSID"]: g for g in lifted}
        n_coord_diff = 0
        big_diffs = []
        for ensid in overlap:
            lg = lifted_lookup[ensid]
            ag = abc_genes[ensid]
            if lg["chr"] != ag["chr"]:
                big_diffs.append((ensid, "chrom differs", ag["chr"], lg["chr"]))
                n_coord_diff += 1
            else:
                start_diff = abs(lg["start"] - ag["start"])
                end_diff = abs(lg["end"] - ag["end"])
                if start_diff > 100 or end_diff > 100:
                    big_diffs.append((
                        ensid, f"coords differ by {start_diff}/{end_diff}bp",
                        f"{ag['chr']}:{ag['start']}-{ag['end']}",
                        f"{lg['chr']}:{lg['start']}-{lg['end']}",
                    ))
                    n_coord_diff += 1

        print(f"  Coordinate diffs (>100bp): {n_coord_diff}")
        if big_diffs:
            print(f"\n  Top discrepancies (max 20):")
            for ensid, reason, abc_val, lifted_val in big_diffs[:20]:
                symbol = abc_genes.get(ensid, {}).get("symbol", "?")
                print(f"    {ensid} ({symbol}): {reason}")
                print(f"      ABC:    {abc_val}")
                print(f"      Lifted: {lifted_val}")

    # ---- Cleanup ----
    os.unlink(tmp_in_path)
    os.unlink(tmp_out_path)
    if os.path.exists(tmp_unmapped_path):
        os.unlink(tmp_unmapped_path)


if __name__ == "__main__":
    main()
