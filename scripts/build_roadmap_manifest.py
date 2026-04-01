#!/usr/bin/env python
"""Build a download manifest for Roadmap epigenomes with verified ENCODE mappings.

Queries ENCODE for DNase, H3K27ac, and HiC data across 38 exact/close-matched
Roadmap epigenomes. Selects the best experiment per assay, returns one file per
biological replicate, marks the best (largest) replicate, and saves TSV + JSON.

Usage
-----
python scripts/build_roadmap_manifest.py
python scripts/build_roadmap_manifest.py --dry-run
python scripts/build_roadmap_manifest.py --output data/roadmap_download_manifest
"""

from __future__ import annotations

import argparse
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
# Verified Roadmap → ENCODE mapping (exact + close only)
# ---------------------------------------------------------------------------

ROADMAP_SAMPLES: dict[str, tuple[str, list[str], str]] = {
    # eid: (short_name, [encode_ontology_names], match_quality)
    "E003": ("H1", ["H1"], "exact"),
    "E004": ("H1_BMP4_Mesendoderm", ["mesendoderm"], "exact"),
    "E005": ("H1_BMP4_Trophoblast", ["trophoblast cell"], "close"),
    "E006": ("H1_Mesenchymal_SC", ["mesenchymal stem cell"], "exact"),
    "E007": ("H1_Neuronal_Prog", ["neural progenitor cell"], "close"),
    "E011": ("hESC_CD184_Endoderm", ["endodermal cell"], "exact"),
    "E013": ("hESC_CD56_Mesoderm", ["mesodermal cell"], "close"),
    "E028": ("Breast_vHMEC", ["mammary epithelial cell"], "close"),
    "E037": ("CD4_Memory", ["CD4-positive, alpha-beta memory T cell"], "exact"),
    "E038": ("CD4_Naive", ["naive thymus-derived CD4-positive, alpha-beta T cell"], "exact"),
    "E047": ("CD8_Naive", ["naive thymus-derived CD8-positive, alpha-beta T cell"], "exact"),
    "E055": ("Foreskin_Fibro_1", ["foreskin fibroblast"], "exact"),
    "E056": ("Foreskin_Fibro_2", ["foreskin fibroblast"], "exact"),
    "E057": ("Foreskin_Kerat_2", ["foreskin keratinocyte"], "exact"),
    "E058": ("Foreskin_Kerat_3", ["foreskin keratinocyte"], "exact"),
    "E059": ("Foreskin_Melano_1", ["foreskin melanocyte"], "exact"),
    "E061": ("Foreskin_Melano_3", ["foreskin melanocyte"], "exact"),
    "E065": ("Aorta", ["aorta"], "exact"),
    "E079": ("Esophagus", ["esophagus squamous epithelium"], "close"),
    "E095": ("Left_Ventricle", ["heart left ventricle"], "exact"),
    "E097": ("Ovary", ["ovary"], "exact"),
    "E098": ("Pancreas", ["pancreas"], "exact"),
    "E100": ("Psoas_Muscle", ["psoas muscle"], "exact"),
    "E104": ("Right_Atrium", ["right cardiac atrium"], "exact"),
    "E105": ("Right_Ventricle", ["heart right ventricle"], "exact"),
    "E106": ("Sigmoid_Colon", ["sigmoid colon"], "exact"),
    "E112": ("Thymus", ["thymus"], "exact"),
    "E113": ("Spleen", ["spleen"], "exact"),
    "E114": ("A549", ["A549"], "exact"),
    "E116": ("GM12878", ["GM12878"], "exact"),
    "E117": ("HeLa", ["HeLa-S3", "HeLa"], "exact"),
    "E118": ("HepG2", ["HepG2"], "exact"),
    "E119": ("HMEC", ["mammary epithelial cell"], "close"),
    "E120": ("HSMM", ["skeletal muscle myoblast"], "close"),
    "E122": ("HUVEC", ["endothelial cell of umbilical vein"], "close"),
    "E123": ("K562", ["K562"], "exact"),
    "E127": ("NHEK", ["keratinocyte", "foreskin keratinocyte"], "close"),
    "E128": ("NHLF", ["fibroblast of lung"], "close"),
}

ASSAY_TITLES = {
    "DNase": ["DNase-seq"],
    "H3K27ac": ["Histone ChIP-seq"],
    "HiC": ["intact Hi-C", "Hi-C"],
}

FOURDN_HIC = {
    "HUVEC": "4DNFIAWVDQ8C",
    "HepG2": "4DNFICSTCJQZ",
    "NHEK": "4DNFIL9M97T2",
    "A549": "4DNFID68JQY9",
    "K562": "4DNFITUOMFUQ",
    "GM12878": "4DNFI1UEG1HD",
}

FOURDN_DOWNLOAD = "https://data.4dnucleome.org/files-processed/{acc}/@@download/{acc}.hic"
ENCODE_SEARCH = "https://www.encodeproject.org/search/"


# ---------------------------------------------------------------------------
# ENCODE API
# ---------------------------------------------------------------------------

def _encode_get(url: str, retries: int = 3) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
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


def query_encode_files(biosample_names: list[str], assay: str) -> list[dict]:
    """Search ENCODE for files matching biosample + assay."""
    assay_titles = ASSAY_TITLES[assay]
    all_results: list[dict] = []

    for biosample in biosample_names:
        for assay_title in assay_titles:
            params = {
                "type": "File",
                "status": "released",
                "assembly": "GRCh38",
                "biosample_ontology.term_name": biosample,
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

            url = f"{ENCODE_SEARCH}?{urllib.parse.urlencode(params)}"
            try:
                data = _encode_get(url)
            except urllib.error.HTTPError:
                continue
            except Exception as exc:
                print(f"    [warn] {biosample}/{assay_title}: {exc}", file=sys.stderr)
                continue

            all_results.extend(data.get("@graph", []))

        if all_results:
            break
    return all_results


def select_all_replicates(candidates: list[dict]) -> list[dict]:
    """Pick one file per biological replicate from the best experiment."""
    if not candidates:
        return []

    # Group by experiment
    by_exp: dict[str, list[dict]] = {}
    for f in candidates:
        dataset = f.get("dataset", "")
        if isinstance(dataset, dict):
            dataset = dataset.get("@id", "")
        m = re.search(r"/(ENC\w+)/?$", dataset)
        exp = m.group(1) if m else "unknown"
        by_exp.setdefault(exp, []).append(f)

    # Pick experiment: latest date, then most replicates
    def exp_score(files):
        reps = set()
        max_date = ""
        for f in files:
            reps.update(f.get("biological_replicates", []))
            d = f.get("date_created", "")
            if d > max_date:
                max_date = d
        return (max_date, len(reps))

    best_exp = max(by_exp, key=lambda e: exp_score(by_exp[e]))
    exp_files = by_exp[best_exp]

    # One file per replicate (newest)
    by_rep: dict[int, list[dict]] = {}
    for f in exp_files:
        reps = f.get("biological_replicates", [])
        if len(reps) == 1:
            by_rep.setdefault(reps[0], []).append(f)
        else:
            by_rep.setdefault(0, []).append(f)

    selected = []
    for rep_id in sorted(by_rep):
        files = sorted(by_rep[rep_id], key=lambda x: x.get("date_created", ""), reverse=True)
        selected.append(files[0])
    return selected


def extract_file_info(f: dict) -> dict:
    """Extract key fields from an ENCODE file object."""
    acc = f.get("accession", "")
    ext = f.get("file_format", "bam")
    if ext == "hic":
        url = f"https://www.encodeproject.org/files/{acc}/@@download/{acc}.hic"
    else:
        url = f"https://www.encodeproject.org/files/{acc}/@@download/{acc}.bam"
        ext = "bam"

    dataset = f.get("dataset", "")
    if isinstance(dataset, dict):
        dataset = dataset.get("@id", "")
    m = re.search(r"/(ENC\w+)/?$", dataset)
    experiment = m.group(1) if m else ""

    asv = f.get("analysis_step_version", "")
    if isinstance(asv, dict):
        asv = asv.get("@id", "")
    pipeline = asv.strip("/").split("/")[-1] if isinstance(asv, str) and asv else ""

    award = f.get("award", {})
    phase = award.get("rfa", "") if isinstance(award, dict) else ""

    return {
        "accession": acc,
        "url": url,
        "ext": ext,
        "file_size": f.get("file_size", 0),
        "bio_reps": f.get("biological_replicates", []),
        "experiment": experiment,
        "pipeline_version": pipeline,
        "encode_phase": phase,
        "date_created": f.get("date_created", ""),
        "output_type": f.get("output_type", ""),
        "assembly": f.get("assembly", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Roadmap download manifest from ENCODE.")
    parser.add_argument("--output", default="data/roadmap_download_manifest",
                        help="Output path prefix (default: data/roadmap_download_manifest)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without saving")
    args = parser.parse_args()

    assays = ["DNase", "H3K27ac", "HiC"]
    items: list[dict] = []

    print(f"Building manifest for {len(ROADMAP_SAMPLES)} Roadmap epigenomes × {len(assays)} assays ...\n")

    for eid in sorted(ROADMAP_SAMPLES.keys()):
        short_name, ontology_names, match_quality = ROADMAP_SAMPLES[eid]
        display = f"{eid}_{short_name}"
        print(f"  {display}")

        for assay in assays:
            print(f"    {assay:<10s}", end="", flush=True)

            candidates = query_encode_files(ontology_names, assay)

            # 4DN Hi-C fallback
            if not candidates and assay == "HiC" and short_name in FOURDN_HIC:
                acc = FOURDN_HIC[short_name]
                dest = os.path.join("data", display, "HiC", f"{acc}.hic")
                items.append({
                    "eid": eid,
                    "roadmap_name": short_name,
                    "encode_biosample": ontology_names[0],
                    "match_quality": match_quality,
                    "assay": "HiC",
                    "accession": acc,
                    "url": FOURDN_DOWNLOAD.format(acc=acc),
                    "ext": "hic",
                    "file_size": 0,
                    "bio_rep": 0,
                    "experiment": "",
                    "encode_phase": "",
                    "pipeline_version": "",
                    "date_created": "",
                    "recommended": True,
                    "dest_path": dest,
                    "source": "4DN",
                })
                print(f" {acc} (4DN)")
                continue

            if not candidates:
                items.append({
                    "eid": eid,
                    "roadmap_name": short_name,
                    "encode_biosample": ontology_names[0],
                    "match_quality": match_quality,
                    "assay": assay,
                    "accession": "NOT_FOUND",
                    "url": "",
                    "ext": "",
                    "file_size": 0,
                    "bio_rep": 0,
                    "experiment": "",
                    "encode_phase": "",
                    "pipeline_version": "",
                    "date_created": "",
                    "recommended": False,
                    "dest_path": "",
                    "source": "",
                })
                print(" not found")
                continue

            selected = select_all_replicates(candidates)
            if not selected:
                print(" no replicates")
                continue

            # Mark best (largest file)
            best_idx = max(range(len(selected)), key=lambda i: selected[i].get("file_size", 0))

            for i, f in enumerate(selected):
                info = extract_file_info(f)
                reps = info["bio_reps"]
                rep_str = reps[0] if len(reps) == 1 else 0
                dest = os.path.join("data", display, assay, f"{info['accession']}.{info['ext']}")

                items.append({
                    "eid": eid,
                    "roadmap_name": short_name,
                    "encode_biosample": ontology_names[0],
                    "match_quality": match_quality,
                    "assay": assay,
                    "accession": info["accession"],
                    "url": info["url"],
                    "ext": info["ext"],
                    "file_size": info["file_size"],
                    "bio_rep": rep_str,
                    "experiment": info["experiment"],
                    "encode_phase": info["encode_phase"],
                    "pipeline_version": info["pipeline_version"],
                    "date_created": info["date_created"],
                    "recommended": (i == best_idx),
                    "dest_path": dest,
                    "source": "ENCODE",
                })

            reps_str = ",".join(str(f.get("biological_replicates", [])) for f in selected)
            best_acc = selected[best_idx]["accession"]
            print(f" {len(selected)} reps ({reps_str})  best={best_acc}")

    # Summary
    n_found = sum(1 for it in items if it["accession"] != "NOT_FOUND")
    n_missing = sum(1 for it in items if it["accession"] == "NOT_FOUND")
    n_recommended = sum(1 for it in items if it["recommended"])
    total_size = sum(it["file_size"] for it in items if it["accession"] != "NOT_FOUND")

    def fmt(n):
        if n >= 1024**3: return f"{n/1024**3:.1f} GB"
        if n >= 1024**2: return f"{n/1024**2:.0f} MB"
        return f"{n} B"

    print(f"\n{'═' * 64}")
    print(f"  Files: {n_found} found | {n_missing} not found | {n_recommended} recommended (best per assay)")
    print(f"  Total size: {fmt(total_size)}")
    print(f"  Epigenomes: {len(ROADMAP_SAMPLES)} | Assays: {len(assays)}")
    print(f"{'═' * 64}")

    if args.dry_run:
        print("\nDry run — manifest not saved.")
        return

    # Save TSV
    tsv_path = args.output + ".tsv"
    cols = [
        "eid", "roadmap_name", "encode_biosample", "match_quality", "assay",
        "accession", "url", "ext", "file_size", "bio_rep", "experiment",
        "encode_phase", "pipeline_version", "date_created", "recommended",
        "dest_path", "source",
    ]
    os.makedirs(os.path.dirname(tsv_path) or ".", exist_ok=True)
    with open(tsv_path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for it in items:
            vals = []
            for c in cols:
                v = it.get(c, "")
                if isinstance(v, bool):
                    v = "true" if v else "false"
                elif isinstance(v, list):
                    v = ",".join(str(x) for x in v)
                vals.append(str(v))
            fh.write("\t".join(vals) + "\n")
    print(f"\n  [saved] {tsv_path} ({len(items)} rows)")

    # Save JSON
    json_path = args.output + ".json"
    with open(json_path, "w") as fh:
        json.dump(items, fh, indent=2, default=str)
    print(f"  [saved] {json_path}")


if __name__ == "__main__":
    main()
