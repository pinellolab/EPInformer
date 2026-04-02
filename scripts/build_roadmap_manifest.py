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

FOURDN_FILE = "https://data.4dnucleome.org/files-processed"
FOURDN_SEARCH = "https://data.4dnucleome.org/search/"
FOURDN_DOWNLOAD = FOURDN_FILE + "/{acc}/@@download/{acc}.hic"
ENCODE_SEARCH = "https://www.encodeproject.org/search/"
GENERIC_MATCH_TOKENS = {
    "adult", "alpha", "and", "beta", "cell", "cells", "derived", "donor",
    "human", "line", "negative", "of", "positive", "primary", "the", "with",
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _json_get(url: str, retries: int = 3, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def _encode_get(url: str, retries: int = 3) -> dict:
    return _json_get(url, retries=retries, timeout=60)


def _fourdn_get(url: str, retries: int = 3) -> dict:
    return _json_get(url, retries=retries, timeout=120)


def _url_with_base(base: str, maybe_relative: str) -> str:
    if not maybe_relative:
        return ""
    return urllib.parse.urljoin(base + "/", maybe_relative)


def _probe_public_url(url: str, timeout: int = 20) -> tuple[bool, str]:
    if not url:
        return False, "missing URL"

    for method in ("HEAD", "GET"):
        headers = {}
        if method == "GET":
            headers["Range"] = "bytes=0-0"
        req = urllib.request.Request(url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", resp.getcode())
                if 200 <= status < 400:
                    return True, f"HTTP {status}"
                return False, f"HTTP {status}"
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in (405, 501):
                continue
            return False, f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            return False, f"URL error: {exc.reason}"
        except Exception as exc:
            return False, str(exc)

    return False, "no successful probe"


def _select_accessible_url(candidates: list[tuple[str, str]]) -> tuple[str, list[str]]:
    failures: list[str] = []
    seen: set[str] = set()
    for label, url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        ok, detail = _probe_public_url(url)
        if ok:
            return url, failures
        failures.append(f"{label} -> {detail}")
    return "", failures


def _norm(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    toks = []
    for tok in _norm(text).split():
        if tok in GENERIC_MATCH_TOKENS:
            continue
        toks.append(tok)
    return toks


def _token_match(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 5 and (a.startswith(b) or b.startswith(a)))


def _alias_matches(alias: str, candidate_tokens: list[str]) -> bool:
    alias_tokens = _tokens(alias)
    if not alias_tokens:
        return False
    return all(any(_token_match(tok, cand) for cand in candidate_tokens) for tok in alias_tokens)


def query_fourdn_hic_catalog() -> list[dict]:
    """Fetch the released GRCh38 4DN hic catalog once and filter locally."""
    params = {
        "type": "FileProcessed",
        "status": "released",
        "file_format": "hic",
        "frame": "embedded",
        "format": "json",
        "limit": "all",
    }
    url = f"{FOURDN_SEARCH}?{urllib.parse.urlencode(params)}"
    data = _fourdn_get(url)
    catalog = []
    for item in data.get("@graph", []):
        if item.get("genome_assembly") != "GRCh38":
            continue
        if item.get("file_type") != "contact matrix":
            continue
        exp_type = _norm(item.get("track_and_facet_info", {}).get("experiment_type", ""))
        if "hi c" not in exp_type:
            continue
        catalog.append(item)
    return catalog


def _score_fourdn_candidate(short_name: str, ontology_names: list[str], item: dict) -> float:
    """Score a 4DN hic file for a given Roadmap sample."""
    track = item.get("track_and_facet_info", {})
    biosource = track.get("biosource_name", "")
    condition = track.get("condition", "")
    dataset = track.get("dataset", "")
    exp_type = track.get("experiment_type", "")

    field_tokens = {
        "biosource": _tokens(biosource),
        "condition": _tokens(condition),
        "dataset": _tokens(dataset),
        "experiment_type": _tokens(exp_type),
    }
    all_tokens = []
    for vals in field_tokens.values():
        all_tokens.extend(vals)

    aliases = [short_name, short_name.replace("_", " "), *ontology_names]
    score = 0.0
    matched = False

    if short_name in FOURDN_HIC and item.get("accession") == FOURDN_HIC[short_name]:
        score += 10000.0
        matched = True

    for alias in aliases:
        if not alias:
            continue
        if not _alias_matches(alias, all_tokens):
            continue

        matched = True
        alias_tokens = _tokens(alias)
        alias_score = 200.0 + 50.0 * len(alias_tokens)
        if _alias_matches(alias, field_tokens["biosource"]):
            alias_score += 200.0
        if _alias_matches(alias, field_tokens["condition"]):
            alias_score += 140.0
        if _alias_matches(alias, field_tokens["dataset"]):
            alias_score += 100.0
        score = max(score, alias_score)

    if not matched:
        return 0.0

    if "in situ hi c" in _norm(exp_type):
        score += 25.0
    score += min(item.get("file_size", 0) / (1024 ** 3), 50.0)
    return score


def select_fourdn_hic(short_name: str, ontology_names: list[str], catalog: list[dict]) -> dict | None:
    """Pick the best 4DN hic file for a Roadmap sample."""
    ranked = []
    for item in catalog:
        score = _score_fourdn_candidate(short_name, ontology_names, item)
        if score <= 0:
            continue
        ranked.append((
            score,
            item.get("date_created", ""),
            item.get("file_size", 0),
            item,
        ))

    if not ranked:
        return None

    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return ranked[0][3]


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


def extract_fourdn_file_info(f: dict) -> dict:
    """Extract key fields from a selected 4DN hic file."""
    acc = f.get("accession", "")
    portal_url = _url_with_base(FOURDN_FILE, f.get("href", "")) or FOURDN_DOWNLOAD.format(acc=acc)
    open_data_url = f.get("open_data_url", "")
    selected_url, failures = _select_accessible_url([
        ("open_data_url", open_data_url),
        ("portal_download", portal_url),
    ])
    if failures:
        print(f"      [warn] 4DN URL probe for {acc}: " + "; ".join(failures), file=sys.stderr)

    track = f.get("track_and_facet_info", {})
    return {
        "accession": acc,
        "url": selected_url or open_data_url or portal_url,
        "ext": "hic",
        "file_size": f.get("file_size", 0),
        "bio_reps": [],
        "experiment": "",
        "pipeline_version": track.get("experiment_type", ""),
        "encode_phase": "4DN",
        "date_created": f.get("date_created", ""),
        "output_type": "4DN processed",
        "assembly": f.get("genome_assembly", ""),
        "biosource_name": track.get("biosource_name", ""),
        "condition": track.get("condition", ""),
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _fmt_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n} B"


def _report_rows(cell_items: dict[str, list[dict]], column_key: str) -> list[dict]:
    """Return rows that should appear in a given HTML report column."""
    if column_key == "4DN":
        return [
            it for it in cell_items.get("HiC", [])
            if it["accession"] != "NOT_FOUND" and it.get("source") == "4DN"
        ]
    if column_key == "HiC":
        return [
            it for it in cell_items.get("HiC", [])
            if it["accession"] != "NOT_FOUND" and it.get("source") != "4DN"
        ]
    return [it for it in cell_items.get(column_key, []) if it["accession"] != "NOT_FOUND"]


def _status_badge(cell_items: dict[str, list[dict]]) -> str:
    present = {
        "DNase": bool(_report_rows(cell_items, "DNase")),
        "H3K27ac": bool(_report_rows(cell_items, "H3K27ac")),
        "HiC": bool(_report_rows(cell_items, "HiC") or _report_rows(cell_items, "4DN")),
    }
    if all(present.values()):
        return '<span class="complete">Complete</span>'
    if present["DNase"] and not present["H3K27ac"] and not present["HiC"]:
        return '<span class="dnase-only">DNase only</span>'
    if not any(present.values()):
        return '<span class="unmapped">Unmapped</span>'
    return '<span class="partial">Partial</span>'


def generate_html_report(items: list[dict], path: str) -> None:
    """Generate an HTML summary of the Roadmap manifest."""
    column_order = ["DNase", "H3K27ac", "HiC", "4DN"]

    by_eid: dict[str, dict[str, list[dict]]] = {}
    sample_meta: dict[str, dict[str, str]] = {}
    for item in items:
        eid = item["eid"]
        by_eid.setdefault(eid, {}).setdefault(item["assay"], []).append(item)
        sample_meta[eid] = {
            "roadmap_name": item["roadmap_name"],
            "encode_biosample": item["encode_biosample"],
            "match_quality": item["match_quality"],
        }

    n_found = sum(1 for it in items if it["accession"] != "NOT_FOUND")
    n_missing = sum(1 for it in items if it["accession"] == "NOT_FOUND")
    n_recommended = sum(1 for it in items if it["recommended"])
    total_size = sum(it["file_size"] for it in items if it["accession"] != "NOT_FOUND")

    assay_cards = []
    for label, cls, rows in (
        ("DNase", "dnase", [it for it in items if it["assay"] == "DNase" and it["accession"] != "NOT_FOUND"]),
        ("H3K27ac", "h3k", [it for it in items if it["assay"] == "H3K27ac" and it["accession"] != "NOT_FOUND"]),
        ("HiC", "hic", [it for it in items if it["assay"] == "HiC" and it["accession"] != "NOT_FOUND" and it.get("source") != "4DN"]),
        ("4DN", "hic", [it for it in items if it["assay"] == "HiC" and it["accession"] != "NOT_FOUND" and it.get("source") == "4DN"]),
    ):
        assay_cards.append(
            f"""  <div class="card card-assay card-{cls}">
    <b>{len(rows)}</b>{label}<br>
    <small>{len(rows)} files | {_fmt_size(sum(it["file_size"] for it in rows))}</small>
  </div>"""
        )

    rows_html = []
    for eid in sorted(by_eid):
        meta = sample_meta[eid]
        cell_items = by_eid[eid]
        mq_class = "mq-exact" if meta["match_quality"] == "exact" else "mq-close"
        row = [
            "<tr>",
            f'  <td class="eid">{eid}</td>',
            f'  <td>{meta["roadmap_name"].replace("_", " ")}</td>',
            f'  <td class="enc-bio">{meta["encode_biosample"]}</td>',
            f'  <td><span class="mq {mq_class}">{meta["match_quality"]}</span></td>',
        ]

        for column_key in column_order:
            assay_rows = _report_rows(cell_items, column_key)
            if not assay_rows:
                row.append('  <td class="missing">-</td>')
                continue

            chunks = []
            for it in assay_rows:
                source = it.get("source", "")
                link = (
                    f'https://data.4dnucleome.org/files-processed/{it["accession"]}/'
                    if source == "4DN"
                    else f'https://www.encodeproject.org/files/{it["accession"]}/'
                )
                best = ' <span class="best">best</span>' if it["recommended"] else ""
                if source == "4DN":
                    phase = ' <span class="phase phase-4dn">4DN</span>'
                    exp = ""
                else:
                    phase_num = str(it.get("encode_phase", "")).replace("ENCODE", "")
                    phase_class = f"phase-{phase_num}" if phase_num in ("2", "3", "4") else ""
                    phase = f' <span class="phase {phase_class}">{it.get("encode_phase", "")}</span>' if phase_class else ""
                    exp = f' <span class="exp">{it.get("experiment", "")}</span>' if it.get("experiment", "") else ""
                detail = f'rep {it.get("bio_rep", 0) or "-"} | {_fmt_size(it.get("file_size", 0))}'
                chunks.append(
                    f'<div class="file-row"><a href="{link}" target="_blank">{it["accession"]}</a>'
                    f'{best} <span class="detail">{detail}</span>{phase}{exp}</div>'
                )
            row.append(f'  <td class="found">{"".join(chunks)}</td>')

        row.append(f"  <td>{_status_badge(cell_items)}</td>")
        row.append("</tr>")
        rows_html.append("\n".join(row))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Roadmap ENCODE Download Manifest</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2em; background: #fafafa; color: #333; }}
  h1 {{ color: #1a1a2e; margin-bottom: 0.3em; }}
  h2 {{ color: #555; font-size: 1em; font-weight: 400; margin-top: 0; }}
  .cards {{ display: flex; gap: 1em; flex-wrap: wrap; margin-bottom: 1.5em; }}
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1em 1.5em;
           min-width: 150px; text-align: center; }}
  .card b {{ display: block; font-size: 2em; color: #1a1a2e; }}
  .card small {{ color: #888; }}
  .card-assay {{ border-left: 4px solid; }}
  .card-dnase {{ border-color: #3b82f6; }}
  .card-h3k {{ border-color: #22c55e; }}
  .card-hic {{ border-color: #a855f7; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px;
           overflow: hidden; font-size: 0.83em; }}
  th {{ background: #1a1a2e; color: #fff; padding: 8px 10px; text-align: left;
       font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.5px;
       position: sticky; top: 0; z-index: 10; }}
  th.assay-dnase {{ background: #1e40af; }}
  th.assay-h3k {{ background: #166534; }}
  th.assay-hic {{ background: #6b21a8; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top; }}
  tr:hover {{ background: #f5f5ff; }}
  .eid {{ font-weight: 700; font-family: monospace; white-space: nowrap; }}
  .enc-bio {{ color: #666; font-size: 0.88em; max-width: 200px; }}
  a {{ color: #2563eb; text-decoration: none; font-weight: 600; font-size: 0.92em; }}
  a:hover {{ text-decoration: underline; }}
  .file-row {{ margin-bottom: 3px; line-height: 1.5; }}
  .detail {{ color: #888; font-size: 0.82em; }}
  .exp {{ color: #aaa; font-size: 0.78em; }}
  .missing {{ color: #ddd; text-align: center; }}
  .found {{ min-width: 200px; }}
  .mq {{ font-size: 0.72em; padding: 2px 8px; border-radius: 4px; font-weight: 700; }}
  .mq-exact {{ background: #22c55e; color: #fff; }}
  .mq-close {{ background: #3b82f6; color: #fff; }}
  .best {{ background: #22c55e; color: #fff; font-size: 0.68em; padding: 1px 5px;
           border-radius: 3px; font-weight: 700; vertical-align: middle; }}
  .phase {{ font-size: 0.68em; padding: 1px 5px; border-radius: 3px; font-weight: 700; margin-left: 2px; }}
  .phase-4 {{ background: #2563eb; color: #fff; }}
  .phase-3 {{ background: #f59e0b; color: #fff; }}
  .phase-2 {{ background: #9ca3af; color: #fff; }}
  .phase-4dn {{ background: #a855f7; color: #fff; }}
  .complete {{ background: #22c55e; color: #fff; font-size: 0.72em; padding: 2px 8px;
               border-radius: 4px; font-weight: 700; }}
  .partial {{ background: #f59e0b; color: #fff; font-size: 0.72em; padding: 2px 8px;
              border-radius: 4px; font-weight: 700; }}
  .dnase-only {{ background: #3b82f6; color: #fff; font-size: 0.72em; padding: 2px 8px;
                 border-radius: 4px; font-weight: 700; }}
  .unmapped {{ background: #6b7280; color: #fff; font-size: 0.72em; padding: 2px 8px;
               border-radius: 4px; font-weight: 700; }}
  input[type=text] {{ padding: 6px 12px; border: 1px solid #ddd; border-radius: 6px;
                      width: 300px; margin-bottom: 1em; font-size: 0.9em; }}
  footer {{ margin-top: 2em; color: #999; font-size: 0.8em; }}
</style>
<script>
function filterTable() {{
  const q = document.getElementById('filter').value.toLowerCase();
  for (const row of document.querySelectorAll('#tbl tbody tr')) {{
    row.style.display = row.innerText.toLowerCase().includes(q) ? '' : 'none';
  }}
}}
</script>
</head>
<body>
<h1>Roadmap ENCODE Download Manifest</h1>
<h2>38 verified epigenomes x 3 assays — all biological replicates, best marked</h2>
<div class="cards">
  <div class="card"><b>38</b>Epigenomes<br><small>exact + close match</small></div>
  <div class="card"><b>{n_found}</b>Files<br><small>{n_missing} not found</small></div>
  <div class="card"><b>{_fmt_size(total_size)}</b>Total size<br><small>{n_recommended} recommended</small></div>
{chr(10).join(assay_cards)}
</div>
<input type="text" id="filter" placeholder="Filter by EID, name, biosample..." oninput="filterTable()">
<table id="tbl">
<thead>
<tr>
  <th>EID</th>
  <th>Roadmap Name</th>
  <th>ENCODE Biosample</th>
  <th>Match</th>
  <th class="assay-dnase">DNase (unfiltered BAM)</th>
  <th class="assay-h3k">H3K27ac (unfiltered BAM)</th>
  <th class="assay-hic">Hi-C (.hic)</th>
  <th class="assay-hic">4DN (.hic)</th>
  <th>Status</th>
</tr>
</thead>
<tbody>
{chr(10).join(rows_html)}
</tbody>
</table>
<footer>Generated by scripts/build_roadmap_manifest.py</footer>
</body>
</html>
"""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(html)
    print(f"  [saved] {path}")


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
    fourdn_catalog = query_fourdn_hic_catalog()

    print(f"Building manifest for {len(ROADMAP_SAMPLES)} Roadmap epigenomes × {len(assays)} assays ...\n")
    print(f"Loaded {len(fourdn_catalog)} released GRCh38 4DN hic files.\n")

    for eid in sorted(ROADMAP_SAMPLES.keys()):
        short_name, ontology_names, match_quality = ROADMAP_SAMPLES[eid]
        display = f"{eid}_{short_name}"
        print(f"  {display}")

        for assay in assays:
            print(f"    {assay:<10s}", end="", flush=True)

            candidates = query_encode_files(ontology_names, assay)
            fourdn_match = None
            if assay == "HiC":
                fourdn_match = select_fourdn_hic(short_name, ontology_names, fourdn_catalog)

            # 4DN Hi-C fallback
            if not candidates and assay == "HiC" and fourdn_match:
                info = extract_fourdn_file_info(fourdn_match)
                acc = info["accession"]
                dest = os.path.join("data", display, "HiC", f"{acc}.hic")
                items.append({
                    "eid": eid,
                    "roadmap_name": short_name,
                    "encode_biosample": ontology_names[0],
                    "match_quality": match_quality,
                    "assay": "HiC",
                    "accession": info["accession"],
                    "url": info["url"],
                    "ext": info["ext"],
                    "file_size": info["file_size"],
                    "bio_rep": 0,
                    "experiment": info["experiment"],
                    "encode_phase": info["encode_phase"],
                    "pipeline_version": info["pipeline_version"],
                    "date_created": info["date_created"],
                    "recommended": True,
                    "dest_path": dest,
                    "source": "4DN",
                })
                biosource = info["biosource_name"] or "?"
                print(f" {acc} (4DN; biosource={biosource})")
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
            msg = f" {len(selected)} reps ({reps_str})  best={best_acc}"
            if assay == "HiC" and fourdn_match:
                info = extract_fourdn_file_info(fourdn_match)
                acc = info["accession"]
                dest = os.path.join("data", display, "HiC", f"{acc}.hic")
                items.append({
                    "eid": eid,
                    "roadmap_name": short_name,
                    "encode_biosample": ontology_names[0],
                    "match_quality": match_quality,
                    "assay": "HiC",
                    "accession": info["accession"],
                    "url": info["url"],
                    "ext": info["ext"],
                    "file_size": info["file_size"],
                    "bio_rep": 0,
                    "experiment": info["experiment"],
                    "encode_phase": info["encode_phase"],
                    "pipeline_version": info["pipeline_version"],
                    "date_created": info["date_created"],
                    "recommended": True,
                    "dest_path": dest,
                    "source": "4DN",
                })
                msg += f" | 4DN={acc}"
            print(msg)

    # Summary
    n_found = sum(1 for it in items if it["accession"] != "NOT_FOUND")
    n_missing = sum(1 for it in items if it["accession"] == "NOT_FOUND")
    n_recommended = sum(1 for it in items if it["recommended"])
    total_size = sum(it["file_size"] for it in items if it["accession"] != "NOT_FOUND")

    print(f"\n{'═' * 64}")
    print(f"  Files: {n_found} found | {n_missing} not found | {n_recommended} recommended (best per assay)")
    print(f"  Total size: {_fmt_size(total_size)}")
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

    html_path = args.output + ".html"
    generate_html_report(items, html_path)


if __name__ == "__main__":
    main()
