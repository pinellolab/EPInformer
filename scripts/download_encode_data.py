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

# Parallel download (4 files at once)
python scripts/download_encode_data.py --from-manifest data/encode_manifest.json --parallel 4

# Generate aria2c input for fast multi-connection download
python scripts/download_encode_data.py --from-manifest data/roadmap_download_manifest.json --aria2c

# Same, but place files under data/roadmap_encode_downloads/... (rebase JSON paths)
python scripts/download_encode_data.py --from-manifest data/roadmap_download_manifest.json \\
    --manifest-root data/roadmap_encode_downloads --aria2c
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
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
    "E016": ("HUES64", []),  # HUES64 on ENCODE but has 0 DNase BAMs
    "E024": ("ES_UCSF4", []),  # ESC line (not T cell); no ENCODE match
    "E027": ("Breast_Myoepithelial", []),  # no match on ENCODE
    "E028": ("Breast_vHMEC", ["mammary epithelial cell"]),
    "E037": ("CD4_Memory", ["CD4-positive, alpha-beta memory T cell", "CD4-positive, alpha-beta T cell"]),
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

DEFAULT_OUTPUT_DIR = "downloads"

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
# Manifest I/O
# ---------------------------------------------------------------------------

MANIFEST_TSV_COLUMNS = [
    "cell_type", "assay", "accession", "url", "ext", "file_size",
    "bio_reps", "experiment", "encode_phase", "pipeline_version",
    "date_created", "recommended", "dest_path", "output_type", "assembly",
]


def save_manifest(items: list[DownloadItem], path: str) -> None:
    """Save download manifest as TSV + JSON."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # TSV
    tsv_path = path if path.endswith(".tsv") else path + ".tsv"
    with open(tsv_path, "w") as fh:
        fh.write("\t".join(MANIFEST_TSV_COLUMNS) + "\n")
        for item in items:
            vals = []
            for col in MANIFEST_TSV_COLUMNS:
                v = getattr(item, col, "")
                if isinstance(v, list):
                    v = ",".join(str(x) for x in v)
                elif isinstance(v, bool):
                    v = "true" if v else "false"
                vals.append(str(v))
            fh.write("\t".join(vals) + "\n")
    print(f"  [manifest] {tsv_path} ({len(items)} entries)")

    # JSON
    json_path = path.replace(".tsv", "") + ".json" if path.endswith(".tsv") else path + ".json"
    with open(json_path, "w") as fh:
        data = []
        for item in items:
            d = {k: v for k, v in item.__dict__.items()}
            data.append(d)
        json.dump(data, fh, indent=2, default=str)
    print(f"  [manifest] {json_path}")


def load_manifest(path: str) -> list[DownloadItem]:
    """Load download manifest from JSON. Handles both download_encode and build_roadmap formats."""
    with open(path) as fh:
        data = json.load(fh)
    items = []
    for d in data:
        # Support both formats: cell_type (download_encode) or eid+roadmap_name (build_roadmap)
        cell_type = d.get("cell_type", "")
        if not cell_type:
            eid = d.get("eid", "")
            name = d.get("roadmap_name", "")
            cell_type = f"{eid}_{name}" if eid else name

        item = DownloadItem(
            cell_type=cell_type,
            assay=d.get("assay", ""),
            accession=d.get("accession", ""),
            url=d.get("url", ""),
            ext=d.get("ext", ""),
        )
        for k in ("file_size", "date_created", "experiment", "bio_reps",
                   "assembly", "output_type", "dest_path", "exists",
                   "recommended", "pipeline_version", "encode_phase"):
            if k in d:
                setattr(item, k, d[k])
        # bio_rep (singular, from roadmap manifest) → bio_reps (list)
        if "bio_rep" in d and not item.bio_reps:
            v = d["bio_rep"]
            item.bio_reps = [v] if v else []
        # Re-check exists on disk
        if item.dest_path:
            item.exists = os.path.exists(item.dest_path)
        items.append(item)
    print(f"Loaded {len(items)} items from {path}")
    return items


def rebase_manifest_dest_paths(
    items: list[DownloadItem],
    new_root: str,
    strip_prefix: str = "data",
) -> None:
    """Rewrite dest_path to live under new_root, dropping strip_prefix from manifest paths."""
    strip_prefix = strip_prefix.rstrip(os.sep) or strip_prefix
    n = 0
    for it in items:
        if not it.dest_path:
            continue
        try:
            rel = os.path.relpath(it.dest_path, strip_prefix)
        except ValueError:
            continue
        if rel.startswith(".." + os.sep) or rel == "..":
            print(
                f"  [manifest-root] skip rebase (not under {strip_prefix}): {it.dest_path}",
                file=sys.stderr,
            )
            continue
        it.dest_path = os.path.normpath(os.path.join(new_root, rel))
        it.exists = os.path.exists(it.dest_path)
        n += 1
    print(f"Rebased {n} dest_path values under {new_root!r} (prefix {strip_prefix!r})")


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
# Download log
# ---------------------------------------------------------------------------

class DownloadLog:
    """Thread-safe download log that writes a TSV log and tracks failed items."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"download_{ts}.log")
        self.failed_path = os.path.join(log_dir, f"download_{ts}_failed.json")
        self._lock = threading.Lock()
        self._failed: list[dict] = []
        self._ok = 0
        self._skip = 0
        self._fail = 0
        # Write header
        with open(self.log_path, "w") as fh:
            fh.write("timestamp\tstatus\taccession\tcell_type\tassay\tfile_size\tdest_path\terror\n")

    def record(self, status: str, item: DownloadItem, error: str = ""):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            if status == "ok":
                self._ok += 1
            elif status == "skip":
                self._skip += 1
            elif status == "fail":
                self._fail += 1
                self._failed.append(item.__dict__)
            with open(self.log_path, "a") as fh:
                fh.write(f"{ts}\t{status}\t{item.accession}\t{item.cell_type}\t"
                         f"{item.assay}\t{item.file_size}\t{item.dest_path}\t{error}\n")

    def finish(self):
        if self._failed:
            with open(self.failed_path, "w") as fh:
                json.dump(self._failed, fh, indent=2, default=str)
            print(f"\n  [log] {self._fail} failed — retry with: "
                  f"--from-manifest {self.failed_path}")
        print(f"  [log] {self.log_path}  "
              f"(ok={self._ok}, skip={self._skip}, fail={self._fail})")

    @property
    def n_failed(self) -> int:
        return self._fail


# ---------------------------------------------------------------------------
# Parallel download
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _download_one(
    idx: int, total: int, item: DownloadItem, force: bool, no_metadata: bool,
    log: DownloadLog | None = None,
) -> bool:
    """Download a single item (called from thread pool)."""
    tag = f"[{idx}/{total}]"
    if item.exists and not force:
        with _print_lock:
            print(f"{tag} [skip] {item.dest_path}")
        if log:
            log.record("skip", item)
        return True

    with _print_lock:
        print(f"{tag} [start] {item.cell_type}/{item.assay}: {item.accession} "
              f"({_fmt_size(item.file_size)})")

    os.makedirs(os.path.dirname(item.dest_path), exist_ok=True)
    part = item.dest_path + ".part"
    req = urllib.request.Request(item.url)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            downloaded = 0
            chunk_size = 1024 * 1024

            with open(part, "wb") as fh:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)

        os.rename(part, item.dest_path)
        with _print_lock:
            print(f"{tag} [done]  {item.accession} → {item.dest_path}")
        if log:
            log.record("ok", item)
    except Exception as exc:
        with _print_lock:
            print(f"{tag} [error] {item.accession}: {exc}", file=sys.stderr)
        if os.path.exists(part):
            os.remove(part)
        if log:
            log.record("fail", item, str(exc))
        return False

    if not no_metadata:
        meta = fetch_metadata(item.accession)
        meta["cell_type"] = item.cell_type
        meta["assay_short"] = item.assay
        meta["recommended"] = item.recommended
        meta["biological_replicates_in_file"] = item.bio_reps
        save_metadata(meta, item.dest_path)

    return True


def download_parallel(
    items: list[DownloadItem], n_workers: int, force: bool, no_metadata: bool,
    log: DownloadLog | None = None,
) -> None:
    """Download items using a thread pool."""
    downloadable = [
        it for it in items
        if it.accession not in ("NOT_FOUND", "NO_MAPPING")
    ]
    total = len(downloadable)
    print(f"\nDownloading {total} files with {n_workers} parallel workers ...\n")

    ok = 0
    fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_download_one, i + 1, total, it, force, no_metadata, log): it
            for i, it in enumerate(downloadable)
        }
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                ok += 1
            else:
                fail += 1

    print(f"\nParallel download complete: {ok} succeeded, {fail} failed.")


# ---------------------------------------------------------------------------
# aria2c support
# ---------------------------------------------------------------------------

def generate_aria2c_input(items: list[DownloadItem], output_path: str) -> str:
    """Write an aria2c input file and return its path."""
    downloadable = [
        it for it in items
        if it.accession not in ("NOT_FOUND", "NO_MAPPING")
    ]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        for it in downloadable:
            dest_dir = os.path.dirname(it.dest_path)
            dest_file = os.path.basename(it.dest_path)
            fh.write(f"{it.url}\n")
            fh.write(f"  dir={dest_dir}\n")
            fh.write(f"  out={dest_file}\n")
    print(f"aria2c input file: {output_path} ({len(downloadable)} files)")
    return output_path


def run_aria2c(
    input_path: str,
    connections: int = 4,
    parallel: int = 3,
    connect_timeout: int = 60,
    stall_timeout: int = 600,
) -> None:
    """Run aria2c with the given input file."""
    aria2c = shutil.which("aria2c")
    if not aria2c:
        print("aria2c not found on PATH. Install it or use the input file manually:")
        print(f"  aria2c -i {input_path} -x {connections} -j {parallel} -c "
              f"--auto-file-renaming=false "
              f"--connect-timeout={connect_timeout} --timeout={stall_timeout}")
        return

    cmd = [
        aria2c,
        f"-i{input_path}",
        f"-x{connections}",
        f"-j{parallel}",
        f"--connect-timeout={connect_timeout}",
        f"--timeout={stall_timeout}",
        "-c",  # resume
        "--auto-file-renaming=false",
        "--console-log-level=notice",
    ]
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


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
        default=DEFAULT_OUTPUT_DIR,
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
    p.add_argument(
        "--save-manifest",
        default=None,
        metavar="PATH",
        help="Save download manifest as TSV + JSON (e.g., --save-manifest data/encode_manifest)",
    )
    p.add_argument(
        "--from-manifest",
        default=None,
        metavar="PATH",
        help="Load manifest JSON instead of querying ENCODE API (e.g., --from-manifest data/encode_manifest.json)",
    )
    p.add_argument(
        "--manifest-root",
        default=None,
        metavar="DIR",
        help="With --from-manifest: place files under DIR by rebasing each dest_path: "
             "strip --manifest-path-prefix from the JSON path, then join DIR. "
             "Example: JSON data/E003_H1/DNase/x.bam with "
             "--manifest-root data/roadmap_encode_downloads → "
             "data/roadmap_encode_downloads/E003_H1/DNase/x.bam",
    )
    p.add_argument(
        "--manifest-path-prefix",
        default="data",
        metavar="PREFIX",
        help="Directory prefix stripped from manifest dest_path before joining --manifest-root "
             "(default: %(default)s).",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Download N files in parallel using threads (default: 1 = sequential)",
    )
    p.add_argument(
        "--aria2c",
        nargs="?",
        const="auto",
        default=None,
        metavar="INPUT_FILE",
        help="Generate aria2c input file and run aria2c. Optionally specify output path "
             "(default: data/aria2c_download.txt). Much faster than Python downloads.",
    )
    p.add_argument(
        "--aria2c-connections",
        type=int,
        default=4,
        metavar="N",
        help="Connections per file for aria2c -x (default: 4)",
    )
    p.add_argument(
        "--aria2c-parallel",
        type=int,
        default=3,
        metavar="N",
        help="Parallel file downloads for aria2c -j (default: 3)",
    )
    p.add_argument(
        "--aria2c-connect-timeout",
        type=int,
        default=60,
        metavar="SEC",
        help="aria2c --connect-timeout in seconds (default: %(default)s)",
    )
    p.add_argument(
        "--aria2c-timeout",
        type=int,
        default=600,
        metavar="SEC",
        help="aria2c --timeout: no-data stall limit in seconds (default: %(default)s)",
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

    if args.manifest_root and not args.from_manifest:
        print("Error: --manifest-root requires --from-manifest", file=sys.stderr)
        sys.exit(1)

    if args.from_manifest:
        # Load from saved manifest — skip API queries
        items = load_manifest(args.from_manifest)
        if args.manifest_root:
            rebase_manifest_dest_paths(
                items,
                args.manifest_root,
                args.manifest_path_prefix,
            )
    else:
        # Query ENCODE API
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

    if args.save_manifest:
        save_manifest(items, args.save_manifest)

    if args.report:
        generate_html_report(items, args.report)

    if args.dry_run:
        print("Dry run — no files downloaded.")
        return

    # aria2c mode
    if args.aria2c is not None:
        if args.aria2c != "auto":
            input_path = args.aria2c
        elif args.manifest_root:
            input_path = os.path.join(args.manifest_root, "aria2c_download.txt")
        else:
            input_path = "data/aria2c_download.txt"
        generate_aria2c_input(items, input_path)
        run_aria2c(
            input_path,
            args.aria2c_connections,
            args.aria2c_parallel,
            connect_timeout=args.aria2c_connect_timeout,
            stall_timeout=args.aria2c_timeout,
        )
        return

    # Create download log
    log = DownloadLog(os.path.join(args.output_dir, ".logs"))

    # Parallel mode
    if args.parallel > 1:
        download_parallel(items, args.parallel, args.force, args.no_metadata, log)
        log.finish()
        return

    # Sequential download (default)
    downloadable = [
        it for it in items
        if it.accession not in ("NOT_FOUND", "NO_MAPPING")
    ]
    for i, item in enumerate(downloadable, 1):
        if item.exists and not args.force:
            print(f"[{i}/{len(downloadable)}] [skip] {item.dest_path}")
            log.record("skip", item)
            continue

        print(f"\n[{i}/{len(downloadable)}] {item.cell_type}/{item.assay}: {item.accession}")
        ok = download_file(item.url, item.dest_path, force=args.force)

        if ok:
            log.record("ok", item)
            if not args.no_metadata:
                meta = fetch_metadata(item.accession)
                meta["cell_type"] = item.cell_type
                meta["assay_short"] = item.assay
                meta["recommended"] = item.recommended
                meta["biological_replicates_in_file"] = item.bio_reps
                save_metadata(meta, item.dest_path)
        else:
            log.record("fail", item)

    log.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
