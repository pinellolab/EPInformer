#!/usr/bin/env python
"""Download unfiltered-alignment BAM and Hi-C files from ENCODE / 4DN.

Queries the ENCODE REST API to find the best available files for each
(cell_type, assay) pair, downloads them, and saves per-file metadata JSON.

Cell types are read from data/cell_line_list.txt by default.

Usage
-----
# Dry run — show what would be downloaded for all cell lines
python scripts/download_encode_data.py --dry-run

# Download specific cell types
python scripts/download_encode_data.py --cell-types HepG2,H1,NHEK

# Download all replicates (one BAM per biological replicate)
python scripts/download_encode_data.py --all-replicates --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODE_SEARCH = "https://www.encodeproject.org/search/"
ENCODE_FILE = "https://www.encodeproject.org/files"

# Cell line list file: "EID,NAME" per line (e.g., "E003,H1_Cell_Line")
CELL_LINE_LIST = "data/cell_line_list.txt"

# Biosample names as they appear in ENCODE ontology (term_name).
# Keys are our short names (matching cell_line_list.txt after normalization).
# Values are alternative ontology names to try in order.
BIOSAMPLE_NAMES: dict[str, list[str]] = {
    "H1": ["H1"],
    "A549": ["A549"],
    "GM12878": ["GM12878"],
    "HeLa": ["HeLa-S3", "HeLa"],
    "HepG2": ["HepG2"],
    "HMEC": ["mammary epithelial cell"],
    "HSMM": ["skeletal muscle myoblast"],
    "HUVEC": ["endothelial cell of umbilical vein"],
    "K562": ["K562"],
    "NHEK": ["keratinocyte", "foreskin keratinocyte"],
    "NHLF": ["fibroblast of lung"],
}

# Full Roadmap epigenome → ENCODE biosample ontology mapping (57 epigenomes).
# Keys are EID (e.g. "E003"). Values are (short_name, [ontology_names]).
ROADMAP_BIOSAMPLE: dict[str, tuple[str, list[str]]] = {
    "E000": ("Universal_Human_Reference", []),  # no ENCODE data
    "E003": ("H1", ["H1"]),
    "E004": ("H1_BMP4_Mesendoderm", ["mesendoderm"]),
    "E005": ("H1_BMP4_Trophoblast", ["trophoblast cell"]),
    "E006": ("H1_Mesenchymal_SC", ["mesenchymal stem cell"]),
    "E007": ("H1_Neuronal_Prog", ["neural progenitor cell"]),
    "E011": ("hESC_CD184_Endoderm", ["endodermal cell"]),
    "E012": ("hESC_CD56_Ectoderm", []),  # no match on ENCODE
    "E013": ("hESC_CD56_Mesoderm", ["mesodermal cell"]),
    "E016": ("HUES64", []),  # no match on ENCODE
    "E024": ("CD4_CD25_Treg", ["CD4-positive, CD25-positive, alpha-beta regulatory T cell"]),
    "E027": ("Breast_Myoepithelial", []),  # no match on ENCODE
    "E028": ("Breast_vHMEC", ["mammary epithelial cell"]),
    "E037": ("CD4_Memory", ["CD4-positive, alpha-beta T cell"]),
    "E038": ("CD4_Naive", ["naive thymus-derived CD4-positive, alpha-beta T cell"]),
    "E047": ("CD8_Naive", ["naive thymus-derived CD8-positive, alpha-beta T cell"]),
    "E050": ("CD34_Mobilized", []),  # no match on ENCODE
    "E053": ("Neurosphere_Cortex", []),  # no match on ENCODE
    "E054": ("Neurosphere_GE", ["neural crest cell"]),
    "E055": ("Foreskin_Fibro_1", ["foreskin fibroblast"]),
    "E056": ("Foreskin_Fibro_2", ["foreskin fibroblast"]),
    "E057": ("Foreskin_Kerat_2", ["foreskin keratinocyte"]),
    "E058": ("Foreskin_Kerat_3", ["foreskin keratinocyte"]),
    "E059": ("Foreskin_Melano_1", ["foreskin melanocyte"]),
    "E061": ("Foreskin_Melano_3", ["foreskin melanocyte"]),
    "E062": ("PBMC", []),  # no direct match
    "E065": ("Aorta", ["aorta"]),
    "E066": ("Adult_Liver", ["liver"]),
    "E070": ("Brain_Germinal_Matrix", ["brain"]),
    "E071": ("Brain_Hippocampus", ["brain"]),
    "E079": ("Esophagus", ["esophagus squamous epithelium"]),
    "E082": ("Fetal_Brain", ["brain"]),
    "E084": ("Fetal_Intestine_Lg", ["large intestine"]),
    "E085": ("Fetal_Intestine_Sm", ["small intestine"]),
    "E087": ("Pancreatic_Islets", ["pancreas"]),
    "E094": ("Gastric", ["stomach"]),
    "E095": ("Left_Ventricle", ["heart left ventricle"]),
    "E096": ("Lung", ["lung"]),
    "E097": ("Ovary", ["ovary"]),
    "E098": ("Pancreas", ["pancreas"]),
    "E100": ("Psoas_Muscle", ["psoas muscle"]),
    "E104": ("Right_Atrium", ["right cardiac atrium"]),
    "E105": ("Right_Ventricle", ["heart right ventricle"]),
    "E106": ("Sigmoid_Colon", ["sigmoid colon"]),
    "E109": ("Small_Intestine", ["small intestine"]),
    "E112": ("Thymus", ["thymus"]),
    "E113": ("Spleen", ["spleen"]),
    "E114": ("A549", ["A549"]),
    "E116": ("GM12878", ["GM12878"]),
    "E117": ("HeLa", ["HeLa-S3", "HeLa"]),
    "E118": ("HepG2", ["HepG2"]),
    "E119": ("HMEC", ["mammary epithelial cell"]),
    "E120": ("HSMM", ["skeletal muscle myoblast"]),
    "E122": ("HUVEC", ["endothelial cell of umbilical vein"]),
    "E123": ("K562", ["K562"]),
    "E127": ("NHEK", ["keratinocyte", "foreskin keratinocyte"]),
    "E128": ("NHLF", ["fibroblast of lung"]),
}

# 4DN Hi-C fallback: cell types with Hi-C on 4DN but not ENCODE.
FOURDN_HIC: dict[str, tuple[str, str]] = {
    "HUVEC": ("4DNFIAWVDQ8C", "HUVEC intact Hi-C from 4DN"),
    "HepG2": ("4DNFICSTCJQZ", "HepG2 intact Hi-C from 4DN"),
    "NHEK": ("4DNFIL9M97T2", "NHEK intact Hi-C from 4DN"),
    "A549": ("4DNFID68JQY9", "A549 intact Hi-C from 4DN"),
    "K562": ("4DNFITUOMFUQ", "K562 Hi-C from 4DN"),
    "GM12878": ("4DNFI1UEG1HD", "GM12878 Hi-C from 4DN"),
}

FOURDN_DOWNLOAD = "https://data.4dnucleome.org/files-processed/{acc}/@@download/{acc}.hic"

# Map our assay short names → ENCODE assay_title values
ASSAY_TITLES: dict[str, list[str]] = {
    "DNase": ["DNase-seq"],
    "ATAC": ["ATAC-seq"],
    "H3K27ac": ["Histone ChIP-seq"],
    "HiC": ["intact Hi-C", "Hi-C"],
}

DEFAULT_ASSAYS = ["DNase", "H3K27ac", "ATAC", "HiC"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DownloadItem:
    cell_type: str
    assay: str
    accession: str
    url: str
    ext: str
    file_size: int = 0
    date_created: str = ""
    experiment: str = ""
    bio_reps: list[int] = field(default_factory=list)
    assembly: str = ""
    output_type: str = ""
    dest_path: str = ""
    exists: bool = False
    recommended: bool = False
    pipeline_version: str = ""
    encode_phase: str = ""  # e.g. "ENCODE2", "ENCODE3", "ENCODE4"


# ---------------------------------------------------------------------------
# Cell line list
# ---------------------------------------------------------------------------

def read_cell_line_list(path: str) -> list[str]:
    """Read cell_line_list.txt and return normalized short names."""
    name_map = {
        "H1_Cell_Line": "H1", "H1": "H1",
        "A549": "A549", "GM12878": "GM12878",
        "HELA": "HeLa", "HeLa": "HeLa",
        "HEPG2": "HepG2", "HepG2": "HepG2",
        "HMEC": "HMEC", "HSMM": "HSMM", "HUVEC": "HUVEC",
        "K562": "K562", "NHEK": "NHEK", "NHLF": "NHLF",
    }
    cell_types = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            raw_name = parts[1].strip() if len(parts) > 1 else parts[0].strip()
            short = name_map.get(raw_name, raw_name)
            cell_types.append(short)
    return cell_types


def read_roadmap_list(path: str) -> list[tuple[str, str]]:
    """Read EG.name.txt and return list of (eid, short_name) tuples."""
    result = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            eid = parts[0].strip()
            if not eid.startswith("E"):
                continue
            result.append((eid, parts[1].strip()))
    return result


# ---------------------------------------------------------------------------
# ENCODE API helpers
# ---------------------------------------------------------------------------

def _encode_get(url: str, retries: int = 3) -> dict:
    """GET JSON from ENCODE, with retry on 429 / transient errors."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            raise
    return {}


def query_encode_files(cell_type: str, assay: str) -> list[dict]:
    """Search ENCODE for files matching cell_type + assay.

    Returns raw ENCODE file objects (dicts).
    """
    biosample_variants = BIOSAMPLE_NAMES.get(cell_type, [cell_type])
    assay_titles = ASSAY_TITLES[assay]

    all_results: list[dict] = []

    for biosample_name in biosample_variants:
        for assay_title in assay_titles:
            params: dict[str, str] = {
                "type": "File",
                "status": "released",
                "assembly": "GRCh38",
                "biosample_ontology.term_name": biosample_name,
                "assay_title": assay_title,
                "limit": "50",
                "format": "json",
                "frame": "embedded",
            }

            if assay == "HiC":
                params["file_format"] = "hic"
            else:
                params["file_format"] = "bam"
                params["output_type"] = "unfiltered alignments"

            if assay == "H3K27ac":
                params["target.label"] = "H3K27ac"

            qs = urllib.parse.urlencode(params)
            url = f"{ENCODE_SEARCH}?{qs}"

            try:
                data = _encode_get(url)
            except urllib.error.HTTPError:
                continue
            except Exception as exc:
                print(f"  [warn] API error for {biosample_name}/{assay_title}: {exc}",
                      file=sys.stderr)
                continue

            results = data.get("@graph", [])
            all_results.extend(results)

        if all_results:
            break  # found results for this biosample variant

    return all_results


def _extract_pipeline_version(f: dict) -> str:
    """Extract pipeline step version string from ENCODE file object."""
    asv = f.get("analysis_step_version", "")
    if isinstance(asv, dict):
        asv = asv.get("@id", "")
    if isinstance(asv, str) and asv:
        # e.g. "/analysis-step-versions/dnase-alignment-step-v-1-0/"
        return asv.strip("/").split("/")[-1]
    return ""


def _extract_encode_phase(f: dict) -> str:
    """Extract ENCODE phase (e.g. 'ENCODE4') from embedded award.rfa."""
    award = f.get("award", {})
    if isinstance(award, dict):
        return award.get("rfa", "")
    return ""


def _sort_key(f: dict) -> tuple:
    """Sort key: newest date_created first, then most bio replicates.

    Newer date_created means newer ENCODE pipeline version (files are
    reprocessed and re-uploaded when the pipeline is updated).
    """
    date = f.get("date_created", "")
    n_reps = len(f.get("biological_replicates", []))
    return (date, n_reps)


def select_best_file(candidates: list[dict]) -> dict | None:
    """Pick the single best file (most bio reps, newest)."""
    if not candidates:
        return None
    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0]


def select_all_replicates(candidates: list[dict]) -> list[dict]:
    """Select one BAM per biological replicate from the best experiment.

    Groups files by experiment, picks the experiment with the most replicates
    and latest pipeline version, then returns one file per replicate (newest).
    """
    if not candidates:
        return []

    # Group by experiment
    by_exp: dict[str, list[dict]] = {}
    for f in candidates:
        dataset = f.get("dataset", "")
        exp_match = re.search(r"/(ENC\w+)/?$", dataset)
        exp = exp_match.group(1) if exp_match else "unknown"
        by_exp.setdefault(exp, []).append(f)

    # Pick experiment with latest files and most unique biological replicates
    def exp_score(files: list[dict]) -> tuple:
        all_reps = set()
        max_date = ""
        for f in files:
            all_reps.update(f.get("biological_replicates", []))
            d = f.get("date_created", "")
            if d > max_date:
                max_date = d
        return (max_date, len(all_reps))

    best_exp = max(by_exp, key=lambda e: exp_score(by_exp[e]))
    exp_files = by_exp[best_exp]

    # Group by replicate, pick one file per replicate (newest pipeline)
    by_rep: dict[int, list[dict]] = {}
    for f in exp_files:
        reps = f.get("biological_replicates", [])
        if len(reps) == 1:
            by_rep.setdefault(reps[0], []).append(f)
        else:
            by_rep.setdefault(0, []).append(f)

    selected = []
    for rep_id in sorted(by_rep):
        files = sorted(by_rep[rep_id], key=_sort_key, reverse=True)
        selected.append(files[0])

    return selected


def file_to_download_item(
    cell_type: str, assay: str, f: dict, output_dir: str,
) -> DownloadItem:
    """Convert an ENCODE file dict to a DownloadItem."""
    accession = f["accession"]
    ext = f.get("file_format", "bam")
    if ext == "hic":
        dl_url = f"https://www.encodeproject.org/files/{accession}/@@download/{accession}.hic"
    else:
        dl_url = f"https://www.encodeproject.org/files/{accession}/@@download/{accession}.bam"
        ext = "bam"

    dest = os.path.join(output_dir, cell_type, assay, f"{accession}.{ext}")

    dataset = f.get("dataset", "")
    exp_match = re.search(r"/(ENC\w+)/?$", dataset)
    experiment = exp_match.group(1) if exp_match else ""

    return DownloadItem(
        cell_type=cell_type,
        assay=assay,
        accession=accession,
        url=dl_url,
        ext=ext,
        file_size=f.get("file_size", 0),
        date_created=f.get("date_created", ""),
        experiment=experiment,
        bio_reps=f.get("biological_replicates", []),
        assembly=f.get("assembly", ""),
        output_type=f.get("output_type", ""),
        dest_path=dest,
        exists=os.path.exists(dest),
        pipeline_version=_extract_pipeline_version(f),
        encode_phase=_extract_encode_phase(f),
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def fetch_metadata(accession: str) -> dict[str, Any]:
    """Fetch full file metadata from ENCODE or 4DN."""
    if accession.startswith("4DNF"):
        return {
            "accession": accession,
            "source": "4DN",
            "download_url": FOURDN_DOWNLOAD.format(acc=accession),
            "file_format": "hic",
        }

    url = f"{ENCODE_FILE}/{accession}/?format=json&frame=object"
    try:
        data = _encode_get(url)
    except Exception as exc:
        print(f"  [warn] metadata fetch failed for {accession}: {exc}",
              file=sys.stderr)
        return {"accession": accession, "error": str(exc)}

    dataset = data.get("dataset", "")
    exp_match = re.search(r"/(ENC\w+)/?$", dataset)

    meta = {
        "accession": accession,
        "source": "ENCODE",
        "experiment_accession": exp_match.group(1) if exp_match else "",
        "file_format": data.get("file_format", ""),
        "output_type": data.get("output_type", ""),
        "assembly": data.get("assembly", ""),
        "file_size": data.get("file_size", 0),
        "biological_replicates": data.get("biological_replicates", []),
        "date_created": data.get("date_created", ""),
        "status": data.get("status", ""),
        "assay_title": data.get("assay_title", ""),
        "biosample_ontology": data.get("biosample_ontology", {}),
        "download_url": data.get("href", ""),
        "md5sum": data.get("md5sum", ""),
        "content_md5sum": data.get("content_md5sum", ""),
        "read_count": data.get("read_count", None),
        "mapped_read_length": data.get("mapped_read_length", None),
    }
    return meta


def save_metadata(meta: dict, file_dest: str) -> None:
    """Save metadata JSON alongside the downloaded file."""
    base = os.path.splitext(file_dest)[0]
    meta_path = f"{base}_metadata.json"
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"  [meta] {meta_path}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, dest: str, force: bool = False) -> bool:
    """Stream-download a file with progress."""
    if os.path.exists(dest) and not force:
        print(f"  [skip] {dest} already exists")
        return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"

    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB

            with open(part, "wb") as fh:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        size_gb = downloaded / (1024 ** 3)
                        total_gb = total / (1024 ** 3)
                        print(
                            f"\r  [{pct:5.1f}%] {size_gb:.2f} / {total_gb:.2f} GB",
                            end="", flush=True,
                        )
                    else:
                        size_mb = downloaded / (1024 ** 2)
                        print(f"\r  {size_mb:.1f} MB downloaded", end="", flush=True)

        print()
        os.rename(part, dest)
        print(f"  [done] {dest}")
        return True

    except Exception as exc:
        print(f"\n  [error] download failed: {exc}", file=sys.stderr)
        if os.path.exists(part):
            os.remove(part)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download unfiltered-alignment BAMs and Hi-C files from ENCODE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cell-types",
        default=None,
        help="Comma-separated cell types. If omitted, reads from --cell-list",
    )
    p.add_argument(
        "--cell-list",
        default=CELL_LINE_LIST,
        help="Path to cell line list file (default: %(default)s)",
    )
    p.add_argument(
        "--roadmap",
        default=None,
        metavar="PATH",
        help="Path to Roadmap EG.name.txt to query all 57 epigenomes "
             "(e.g., data/roadmap_expression/.cache/EG.name.txt)",
    )
    p.add_argument(
        "--assays",
        default=",".join(DEFAULT_ASSAYS),
        help="Comma-separated assays (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        default="data",
        help="Base output directory (default: %(default)s)",
    )
    p.add_argument(
        "--all-replicates",
        action="store_true",
        help="Download one BAM per biological replicate (default: single best file)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show download plan without fetching files",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists",
    )
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip metadata fetch/save",
    )
    p.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="Generate HTML summary report (e.g., --report data/encode_download_report.html)",
    )
    return p.parse_args()


def _fmt_size(nbytes: int) -> str:
    if nbytes == 0:
        return "unknown"
    if nbytes >= 1024 ** 3:
        return f"{nbytes / 1024**3:.1f} GB"
    if nbytes >= 1024 ** 2:
        return f"{nbytes / 1024**2:.0f} MB"
    return f"{nbytes} B"


def print_download_plan(items: list[DownloadItem]) -> None:
    """Pretty-print the download plan."""
    current_cell = ""
    total_size = 0
    n_download = 0
    n_skip = 0
    n_missing = 0

    for item in items:
        if item.cell_type != current_cell:
            current_cell = item.cell_type
            print(f"\n  {current_cell}")
            print(f"  {'─' * 60}")

        if item.accession == "NO_MAPPING":
            status = "[no ENCODE mapping]"
            n_missing += 1
        elif item.accession == "NOT_FOUND":
            status = "[not found on ENCODE]"
            n_missing += 1
        elif item.exists:
            status = "[exists, will skip]"
            n_skip += 1
        else:
            status = f"[{_fmt_size(item.file_size)}]"
            total_size += item.file_size
            n_download += 1

        reps = f"reps={item.bio_reps}" if item.bio_reps else ""
        star = " *best*" if item.recommended else ""
        print(
            f"    {item.assay:<8s}  {item.accession:<14s}  "
            f"{status:<24s}  {reps}{star}"
        )
        if item.accession not in ("NOT_FOUND", "NO_MAPPING"):
            print(f"             → {item.dest_path}")
            if item.experiment:
                pv = item.pipeline_version or "?"
                phase = item.encode_phase or "?"
                print(f"               experiment: {item.experiment}  "
                      f"date: {item.date_created[:10] if item.date_created else '?'}  "
                      f"pipeline: {pv}  [{phase}]")

    print(f"\n{'═' * 64}")
    print(f"  To download: {n_download}  |  Skip (exists): {n_skip}  "
          f"|  Not found: {n_missing}")
    print(f"  Estimated size: {_fmt_size(total_size)}")
    print(f"{'═' * 64}\n")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(items: list[DownloadItem], path: str) -> None:
    """Generate an HTML summary report of all download items."""
    from datetime import datetime

    # Group items by cell type
    by_cell: dict[str, list[DownloadItem]] = {}
    for item in items:
        by_cell.setdefault(item.cell_type, []).append(item)

    # Compute stats
    total_size = sum(it.file_size for it in items if it.accession != "NOT_FOUND")
    n_found = sum(1 for it in items if it.accession != "NOT_FOUND")
    n_missing = sum(1 for it in items if it.accession == "NOT_FOUND")

    # Collect all unique assays in order
    assays_seen: list[str] = []
    for it in items:
        if it.assay not in assays_seen:
            assays_seen.append(it.assay)

    rows_html = []
    for cell_type in by_cell:
        cell_items = {it.assay: it for it in by_cell[cell_type]}
        row = f'<tr><td class="cell-type">{cell_type}</td>'
        for assay in assays_seen:
            it = cell_items.get(assay)
            if it is None:
                row += '<td class="na">-</td>'
            elif it.accession == "NO_MAPPING":
                row += '<td class="na">no mapping</td>'
            elif it.accession == "NOT_FOUND":
                row += '<td class="missing">not found</td>'
            else:
                size = _fmt_size(it.file_size)
                reps = ", ".join(str(r) for r in it.bio_reps) if it.bio_reps else "?"
                source = "4DN" if it.accession.startswith("4DNF") else "ENCODE"
                pv = it.pipeline_version or "-"
                date = it.date_created[:10] if it.date_created else "-"
                best_badge = ' <span class="best">best</span>' if it.recommended else ""

                # ENCODE phase badge
                phase = it.encode_phase or ""
                if phase:
                    phase_num = phase.replace("ENCODE", "")
                    phase_cls = f"phase-{phase_num}" if phase_num in ("2", "3", "4") else ""
                    phase_badge = f' <span class="phase {phase_cls}">{phase}</span>'
                else:
                    phase_badge = ""

                # Build ENCODE/4DN link
                if it.accession.startswith("4DNF"):
                    link = f"https://data.4dnucleome.org/files-processed/{it.accession}/"
                else:
                    link = f"https://www.encodeproject.org/files/{it.accession}/"

                exp_link = ""
                if it.experiment:
                    exp_link = (
                        f'<br><a href="https://www.encodeproject.org/experiments/{it.experiment}/"'
                        f' target="_blank">{it.experiment}</a>'
                    )

                row += (
                    f'<td class="found">'
                    f'<a href="{link}" target="_blank">{it.accession}</a>'
                    f'{best_badge}{phase_badge}'
                    f'<br><span class="detail">size: {size} | reps: {reps}</span>'
                    f'<br><span class="detail">pipeline: {pv}</span>'
                    f'<br><span class="detail">date: {date} | src: {source}</span>'
                    f'{exp_link}'
                    f'</td>'
                )
        row += '</tr>'
        rows_html.append(row)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ENCODE Download Report — EPInformer</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2em; background: #fafafa; color: #333; }}
  h1 {{ color: #1a1a2e; }}
  .summary {{ background: #fff; border: 1px solid #ddd; border-radius: 8px;
              padding: 1em 1.5em; margin-bottom: 1.5em; display: inline-block; }}
  .summary span {{ font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px;
           overflow: hidden; }}
  th {{ background: #1a1a2e; color: #fff; padding: 12px 16px; text-align: left;
       font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #eee; vertical-align: top;
       font-size: 0.88em; }}
  tr:hover {{ background: #f5f5ff; }}
  .cell-type {{ font-weight: 700; font-size: 1em; white-space: nowrap; }}
  .found a {{ color: #2563eb; text-decoration: none; font-weight: 600; }}
  .found a:hover {{ text-decoration: underline; }}
  .detail {{ color: #666; font-size: 0.85em; }}
  .missing {{ color: #dc2626; font-style: italic; }}
  .na {{ color: #999; }}
  .best {{ background: #22c55e; color: #fff; font-size: 0.7em; padding: 2px 6px;
           border-radius: 4px; font-weight: 700; vertical-align: middle; }}
  .phase {{ font-size: 0.7em; padding: 2px 6px; border-radius: 4px;
            font-weight: 700; vertical-align: middle; }}
  .phase-4 {{ background: #2563eb; color: #fff; }}
  .phase-3 {{ background: #f59e0b; color: #fff; }}
  .phase-2 {{ background: #9ca3af; color: #fff; }}
  footer {{ margin-top: 2em; color: #999; font-size: 0.8em; }}
</style>
</head>
<body>
<h1>ENCODE Download Report</h1>
<div class="summary">
  <span>{n_found}</span> files found &nbsp;|&nbsp;
  <span>{n_missing}</span> not found &nbsp;|&nbsp;
  <span>{_fmt_size(total_size)}</span> total &nbsp;|&nbsp;
  {len(by_cell)} cell types &nbsp;|&nbsp;
  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
</div>

<table>
<thead>
<tr>
  <th>Cell Type</th>
  {"".join(f"<th>{a}</th>" for a in assays_seen)}
</tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>

<footer>
  Generated by <code>scripts/download_encode_data.py --report</code><br>
  Files use <b>unfiltered alignments</b> (output_type) for BAMs, GRCh38 assembly.<br>
  <span class="best">best</span> = recommended file (deepest sequencing / most replicates).
  <span class="phase phase-4">ENCODE4</span> <span class="phase phase-3">ENCODE3</span> <span class="phase phase-2">ENCODE2</span> = ENCODE project phase.
</footer>
</body>
</html>
"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(html)
    print(f"\nHTML report saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_cell_types(args) -> list[tuple[str, list[str]]]:
    """Return list of (display_name, biosample_ontology_names) pairs."""
    if args.roadmap:
        # Roadmap mode: read EG.name.txt, use ROADMAP_BIOSAMPLE mapping
        entries = read_roadmap_list(args.roadmap)
        result = []
        for eid, raw_name in entries:
            if eid in ROADMAP_BIOSAMPLE:
                short, ontology_names = ROADMAP_BIOSAMPLE[eid]
                display = f"{eid}_{short}"
                result.append((display, ontology_names))
            else:
                result.append((f"{eid}_{raw_name}", []))
        print(f"Read {len(result)} Roadmap epigenomes from {args.roadmap}")
        return result

    if args.cell_types:
        cell_types = [c.strip() for c in args.cell_types.split(",")]
    else:
        if not os.path.exists(args.cell_list):
            print(f"Error: cell line list not found: {args.cell_list}", file=sys.stderr)
            sys.exit(1)
        cell_types = read_cell_line_list(args.cell_list)
        print(f"Read {len(cell_types)} cell types from {args.cell_list}")

    result = []
    for ct in cell_types:
        ontology = BIOSAMPLE_NAMES.get(ct, [ct])
        result.append((ct, ontology))
    return result


def _query_for_sample(
    display_name: str, ontology_names: list[str], assay: str,
    args: argparse.Namespace,
) -> list[DownloadItem]:
    """Query ENCODE for one (sample, assay) pair and return DownloadItems."""
    items: list[DownloadItem] = []

    # No ontology mapping → can't query
    if not ontology_names:
        items.append(DownloadItem(
            cell_type=display_name, assay=assay,
            accession="NO_MAPPING", url="", ext="",
        ))
        return items

    # Temporarily override BIOSAMPLE_NAMES for query
    orig = BIOSAMPLE_NAMES.get(display_name)
    BIOSAMPLE_NAMES[display_name] = ontology_names
    candidates = query_encode_files(display_name, assay)
    if orig is None:
        BIOSAMPLE_NAMES.pop(display_name, None)
    else:
        BIOSAMPLE_NAMES[display_name] = orig

    if not candidates and assay == "HiC":
        # Check 4DN fallback by short name
        short = display_name.split("_", 1)[1] if "_" in display_name else display_name
        if short in FOURDN_HIC:
            acc, desc = FOURDN_HIC[short]
            dl_url = FOURDN_DOWNLOAD.format(acc=acc)
            dest = os.path.join(args.output_dir, display_name, "HiC", f"{acc}.hic")
            items.append(DownloadItem(
                cell_type=display_name, assay="HiC", accession=acc,
                url=dl_url, ext="hic", dest_path=dest,
                exists=os.path.exists(dest),
                output_type="4DN processed",
            ))
            return items

    if not candidates:
        items.append(DownloadItem(
            cell_type=display_name, assay=assay,
            accession="NOT_FOUND", url="", ext="",
        ))
        return items

    if args.all_replicates:
        selected = select_all_replicates(candidates)
        if selected:
            best_idx = max(
                range(len(selected)),
                key=lambda i: selected[i].get("file_size", 0),
            )
        else:
            best_idx = -1
        for i, f in enumerate(selected):
            item = file_to_download_item(display_name, assay, f, args.output_dir)
            item.recommended = (i == best_idx)
            items.append(item)
    else:
        best = select_best_file(candidates)
        item = file_to_download_item(display_name, assay, best, args.output_dir)
        item.recommended = True
        items.append(item)

    return items


def main() -> None:
    args = parse_args()

    samples = _resolve_cell_types(args)
    assays = [a.strip() for a in args.assays.split(",")]

    print(f"Querying ENCODE for {len(samples)} samples × {len(assays)} assays ...\n")

    items: list[DownloadItem] = []

    for display_name, ontology_names in samples:
        for assay in assays:
            print(f"  Searching: {display_name} / {assay} ...", end=" ", flush=True)

            new_items = _query_for_sample(display_name, ontology_names, assay, args)
            items.extend(new_items)

            # Print summary
            if len(new_items) == 1 and new_items[0].accession in ("NOT_FOUND", "NO_MAPPING"):
                label = "no mapping" if new_items[0].accession == "NO_MAPPING" else "not found"
                print(label)
            elif len(new_items) == 1:
                it = new_items[0]
                source = " (4DN)" if it.accession.startswith("4DNF") else ""
                print(f"{it.accession} ({_fmt_size(it.file_size)}, "
                      f"reps={it.bio_reps}){source}")
            else:
                print(f"{len(new_items)} files")

    # Print plan
    print_download_plan(items)

    if args.report:
        generate_html_report(items, args.report)

    if args.dry_run:
        print("Dry run — no files downloaded.")
        return

    # Download
    for item in items:
        if item.accession in ("NOT_FOUND", "NO_MAPPING"):
            continue
        if item.exists and not args.force:
            print(f"[skip] {item.dest_path}")
            continue

        print(f"\n[download] {item.cell_type}/{item.assay}: {item.accession}")
        ok = download_file(item.url, item.dest_path, force=args.force)

        if ok and not args.no_metadata:
            meta = fetch_metadata(item.accession)
            meta["cell_type"] = item.cell_type
            meta["assay_short"] = item.assay
            meta["recommended"] = item.recommended
            meta["biological_replicates_in_file"] = item.bio_reps
            save_metadata(meta, item.dest_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
