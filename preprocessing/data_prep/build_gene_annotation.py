#!/usr/bin/env python3
"""
Build a hg38 gene annotation BED from Roadmap's Ensembl v65 / Gencode v10
gene_info file by lifting over coordinates from hg19 → hg38.

This ensures the gene BED and expression CSV are derived from the same
annotation source (Roadmap Epigenomics).

Requires:
  - liftOver binary (UCSC) on PATH or at --liftover-bin
  - hg19ToHg38.over.chain — either ``hg19ToHg38.over.chain.gz`` or uncompressed
    ``hg19ToHg38.over.chain`` in the cache; auto-downloaded as ``.gz`` if neither exists

Air-gapped / no outbound HTTP: place files under ``{output-dir}/.liftover_cache/`` (or
``--cache-dir``) with the same names the script expects, or pass ``--liftover-bin``,
``--chain-file``, and/or ``--gene-info-file``. The gene_info file is also detected if
already present under ``data/roadmap_expression/.cache/`` (Roadmap download cache).
Otherwise download liftOver, gene_info, and the chain file on a machine with internet,
then copy them into the cache directory.

Usage::

    python preprocessing/data_prep/build_gene_annotation.py \\
        --gene-set pc \\
        --output-dir data/reference/hg38

    python preprocessing/data_prep/build_gene_annotation.py \\
        --gene-set pc_linc \\
        --output-dir data/reference/hg38
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
_ROADMAP_EXPR = "https://egg2.wustl.edu/roadmap/data/byDataType/rna/expression"
_GENE_INFO_FILENAME = "Ensembl_v65.Gencode_v10.ENSG.gene_info"
_GENE_INFO_URL = f"{_ROADMAP_EXPR}/{_GENE_INFO_FILENAME}"
_CHAIN_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"

# liftOver binary download URLs by platform
_LIFTOVER_URLS = {
    ("Darwin", "arm64"):  "https://hgdownload.soe.ucsc.edu/admin/exe/macOSX.arm64/liftOver",
    ("Darwin", "x86_64"): "https://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/liftOver",
    ("Linux", "x86_64"):  "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver",
}

# Gene types to include for each --gene-set option
_GENE_SETS = {
    "pc": {"protein_coding"},
    "pc_linc": {"protein_coding", "lincRNA"},
}


def _print_download_failure(
    url: str,
    dest: str,
    err: BaseException,
    *,
    extra_hint: str | None = None,
) -> None:
    dest_abs = os.path.abspath(dest)
    print("\nERROR: Download failed (no network, firewall, or remote unreachable).", file=sys.stderr)
    print(f"  URL: {url}", file=sys.stderr)
    print(f"  Error: {err}", file=sys.stderr)
    print(
        "\nOffline fix: on a machine with internet, download the URL above, copy the file to:",
        file=sys.stderr,
    )
    print(f"  {dest_abs}", file=sys.stderr)
    print("Then re-run; existing files in the cache are reused without downloading.", file=sys.stderr)
    if os.path.basename(dest_abs) == "liftOver":
        print(
            "\nAfter copying the binary to that path, make it executable, then re-run:\n"
            f"  chmod +x {dest_abs}\n"
            "\nOr point at a liftOver already on this system:\n"
            "  --liftover-bin /path/to/liftOver\n",
            file=sys.stderr,
        )
    elif extra_hint:
        print(extra_hint, file=sys.stderr)
    sys.exit(1)


def _download(url: str, dest: str, *, extra_hint: str | None = None) -> str:
    """Download a file if not already cached."""
    dest = os.path.abspath(os.path.expanduser(dest))
    if os.path.isfile(dest):
        print(f"  Cached: {dest}")
        return dest
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except (urllib.error.URLError, OSError) as e:
        if os.path.isfile(dest):
            try:
                os.unlink(dest)
            except OSError:
                pass
        _print_download_failure(url, dest, e, extra_hint=extra_hint)
    print(f"  Saved: {dest}")
    return dest


def _resolve_chain_file(cache_dir: str, user_chain: str | None) -> str:
    """Return path to hg19→hg38 chain: user path, or cached .gz / uncompressed, or download."""
    if user_chain:
        if not os.path.isfile(user_chain):
            print(f"ERROR: --chain-file not found: {user_chain}", file=sys.stderr)
            sys.exit(1)
        return os.path.abspath(user_chain)
    os.makedirs(cache_dir, exist_ok=True)
    gz = os.path.join(cache_dir, "hg19ToHg38.over.chain.gz")
    plain = os.path.join(cache_dir, "hg19ToHg38.over.chain")
    if os.path.isfile(gz):
        print(f"  Cached: {gz}")
        return gz
    if os.path.isfile(plain):
        print(f"  Cached: {plain}")
        return plain
    return _download(_CHAIN_URL, gz)


def _resolve_gene_info(cache_dir: str, user_path: str | None) -> str:
    """Return path to gene_info: user path, cache file, Roadmap cache dirs, or download."""
    if user_path:
        if not os.path.isfile(user_path):
            print(f"ERROR: --gene-info-file not found: {user_path}", file=sys.stderr)
            sys.exit(1)
        return os.path.abspath(user_path)
    cache_dir = os.path.abspath(cache_dir)
    primary = os.path.join(cache_dir, _GENE_INFO_FILENAME)
    if os.path.isfile(primary):
        print(f"  Cached: {primary}")
        return primary
    # Same file may already exist from build_roadmap_expression / manual staging
    for rel in (
        os.path.join("data", "roadmap_expression", ".cache", _GENE_INFO_FILENAME),
        os.path.join("data", "roadmap_expression", _GENE_INFO_FILENAME),
    ):
        alt = os.path.abspath(rel)
        if os.path.isfile(alt):
            print(f"  Using existing gene_info: {alt}")
            return alt
    return _download(_GENE_INFO_URL, primary)


def _find_liftover(user_bin: str | None, cache_dir: str) -> str:
    """Locate the liftOver binary; auto-download if not found."""
    if user_bin:
        if not os.path.isfile(user_bin):
            print(f"ERROR: --liftover-bin not found: {user_bin}", file=sys.stderr)
            sys.exit(1)
        if not os.access(user_bin, os.X_OK):
            print(f"ERROR: --liftover-bin is not executable: {user_bin}", file=sys.stderr)
            sys.exit(1)
        return os.path.abspath(user_bin)
    found = shutil.which("liftOver")
    if found:
        return found
    # Check cache (fix permissions if copied without +x)
    cached = os.path.join(cache_dir, "liftOver")
    if os.path.isfile(cached):
        if not os.access(cached, os.X_OK):
            try:
                os.chmod(cached, 0o755)
            except OSError as e:
                print(f"ERROR: could not chmod +x cached liftOver: {cached}\n  {e}",
                      file=sys.stderr)
                sys.exit(1)
        if os.access(cached, os.X_OK):
            return cached
        print(
            f"ERROR: cached liftOver is not executable: {cached}\n"
            f"  Run: chmod +x {cached}",
            file=sys.stderr,
        )
        sys.exit(1)
    # Auto-download
    import platform
    key = (platform.system(), platform.machine())
    url = _LIFTOVER_URLS.get(key)
    if not url:
        print(f"ERROR: No liftOver binary available for {key}. "
              f"Download manually from https://hgdownload.soe.ucsc.edu/admin/exe/",
              file=sys.stderr)
        sys.exit(1)
    print(f"  liftOver not found — downloading for {key[0]} {key[1]} ...")
    _download(url, cached)
    os.chmod(cached, 0o755)
    return cached


def _load_gene_info(path: str) -> list[dict]:
    """Parse the Ensembl v65 gene_info file into a list of dicts."""
    genes = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            chrom = parts[1]
            # Skip non-standard chromosomes (patches, haplotypes)
            if "_" in chrom or chrom.startswith("GL") or chrom.startswith("KI"):
                continue
            strand_num = int(parts[4])
            genes.append({
                "ensid": parts[0],
                "chrom": chrom if chrom.startswith("chr") else f"chr{chrom}",
                "start": int(parts[2]),
                "end": int(parts[3]),
                "strand": "+" if strand_num >= 0 else "-",
                "biotype": parts[5],
                "gene_name": parts[6] if parts[6] else parts[0],
            })
    return genes


def _run_liftover(
    genes: list[dict],
    liftover_bin: str,
    chain_file: str,
) -> list[dict]:
    """Liftover gene coordinates from hg19 → hg38 using UCSC liftOver."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f_in:
        in_bed = f_in.name
        for i, g in enumerate(genes):
            # BED format: chr start end name score strand
            f_in.write(f"{g['chrom']}\t{g['start']}\t{g['end']}\t{i}\t0\t{g['strand']}\n")

    out_bed = in_bed + ".lifted"
    unmapped = in_bed + ".unmapped"

    try:
        cmd = [liftover_bin, in_bed, chain_file, out_bed, unmapped]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.isfile(out_bed):
            print("ERROR: liftOver failed or produced no output BED.", file=sys.stderr)
            if result.stdout:
                print(f"liftOver stdout:\n{result.stdout}", file=sys.stderr)
            if result.stderr:
                print(f"liftOver stderr:\n{result.stderr}", file=sys.stderr)
            if not os.path.isfile(out_bed):
                print(f"Missing expected output: {out_bed}", file=sys.stderr)
            sys.exit(1)

        # Parse lifted coordinates
        lifted = {}
        with open(out_bed) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                idx = int(parts[3])
                lifted[idx] = {
                    "chrom": parts[0],
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "strand": parts[5],
                }

        # Count unmapped
        n_unmapped = 0
        if os.path.isfile(unmapped):
            with open(unmapped) as f:
                for line in f:
                    if not line.startswith("#"):
                        n_unmapped += 1

        print(f"  Lifted: {len(lifted)}/{len(genes)} genes "
              f"({n_unmapped} unmapped)")

        # Merge lifted coords back into gene records
        result_genes = []
        for i, g in enumerate(genes):
            if i in lifted:
                g_out = g.copy()
                g_out["chrom"] = lifted[i]["chrom"]
                g_out["start"] = lifted[i]["start"]
                g_out["end"] = lifted[i]["end"]
                g_out["strand"] = lifted[i]["strand"]
                result_genes.append(g_out)

        return result_genes

    finally:
        for f in [in_bed, out_bed, unmapped]:
            if os.path.isfile(f):
                os.unlink(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hg38 gene annotation BED from Roadmap Ensembl v65 gene_info"
    )
    parser.add_argument(
        "--gene-set",
        choices=list(_GENE_SETS.keys()),
        default="pc",
        help="Gene set to include: 'pc' (protein_coding only, ~20K) or "
             "'pc_linc' (protein_coding + lincRNA, ~25K). Default: pc.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/reference/hg38",
        help="Output directory for the BED file. Default: data/reference/hg38",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache downloaded files. Default: {output-dir}/.liftover_cache",
    )
    parser.add_argument(
        "--liftover-bin",
        default=None,
        help="Path to liftOver binary. Default: auto-detect from PATH.",
    )
    parser.add_argument(
        "--chain-file",
        default=None,
        help="Path to hg19ToHg38.over.chain(.gz). Default: auto-download.",
    )
    parser.add_argument(
        "--gene-info-file",
        default=None,
        help=f"Path to {_GENE_INFO_FILENAME}. Default: cache dir or data/roadmap_expression/.cache/, "
             "else download.",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    cache_dir = args.cache_dir or os.path.join(out_dir, ".liftover_cache")
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(out_dir, exist_ok=True)

    # --- Locate tools ---
    liftover_bin = _find_liftover(args.liftover_bin, cache_dir)
    print(f"Using liftOver: {liftover_bin}")

    # --- Download files ---
    print("Downloading reference files ...")
    gene_info_path = _resolve_gene_info(cache_dir, args.gene_info_file)

    chain_file = _resolve_chain_file(cache_dir, args.chain_file)

    # --- Load and filter genes ---
    print("Loading gene annotations ...")
    all_genes = _load_gene_info(gene_info_path)
    print(f"  Total genes parsed: {len(all_genes)}")

    allowed_types = _GENE_SETS[args.gene_set]
    filtered = [g for g in all_genes if g["biotype"] in allowed_types]
    print(f"  After filtering to {allowed_types}: {len(filtered)} genes")

    # --- Liftover hg19 → hg38 ---
    print("Running liftOver hg19 → hg38 ...")
    lifted = _run_liftover(filtered, liftover_bin, chain_file)

    # --- Deduplicate by ENSID (keep first occurrence) ---
    seen = set()
    deduped = []
    for g in lifted:
        if g["ensid"] not in seen:
            seen.add(g["ensid"])
            deduped.append(g)
    print(f"  After dedup: {len(deduped)} unique genes")

    # --- Sort by chromosome and position ---
    chrom_order = {f"chr{c}": i for i, c in enumerate(
        list(range(1, 23)) + ["X", "Y", "M"]
    )}
    deduped.sort(key=lambda g: (chrom_order.get(g["chrom"], 99), g["start"]))

    # --- Write BED ---
    out_name = f"CollapsedGeneBounds.Ensembl_v65.hg38.{args.gene_set}.bed"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w") as f:
        f.write("#chr\tstart\tend\tname\tscore\tstrand\tEnsembl_ID\tgene_type\n")
        for g in deduped:
            f.write(f"{g['chrom']}\t{g['start']}\t{g['end']}\t"
                    f"{g['gene_name']}\t0\t{g['strand']}\t"
                    f"{g['ensid']}\t{g['biotype']}\n")

    print(f"\nDone! {len(deduped)} genes written to {out_path}")

    # --- Summary by type ---
    from collections import Counter
    type_counts = Counter(g["biotype"] for g in deduped)
    for bt, cnt in type_counts.most_common():
        print(f"  {bt}: {cnt}")


if __name__ == "__main__":
    main()
