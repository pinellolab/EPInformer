# ENCODE Data Acquisition Guide

This guide walks through finding, matching, reviewing, and downloading ENCODE data for EPInformer training across Roadmap epigenomes.

## Overview

EPInformer needs three types of ENCODE data per cell type:

| Assay | File type | ENCODE filter | Why |
|-------|-----------|---------------|-----|
| **DNase-seq** | Unfiltered BAM | `output_type=unfiltered alignments` | Accessibility signal for ABC activity scores + MACS2 peak calling |
| **H3K27ac ChIP-seq** | Unfiltered BAM | `target.label=H3K27ac` | Active enhancer signal for ABC activity scores |
| **Hi-C** | `.hic` | `file_format=hic` | Chromatin contact for enhancer-gene linking |

**Important:** Always use **unfiltered alignments** BAMs — the ABC pipeline runs MACS2 with `--keep-dup all` and `pysam.count()` without MAPQ filters. Pre-filtered ENCODE "alignments" BAMs would undercount reads.

## Pipeline: Search → Match → Review → Download

```
Step 1: Search ENCODE          What DNase samples exist?
   ↓                           (369 biosamples with DNase on GRCh38)
Step 2: Match to Roadmap       Map ENCODE biosample ontology → Roadmap EIDs
   ↓                           (38 exact/close matches out of 57)
Step 3: Review mapping          Verify matches, flag issues
   ↓                           (E024 was wrong, tissue terms need broad fallback)
Step 4: Build manifest          Select best experiment + all replicates per assay
   ↓                           (121 files, ~1.1 TB)
Step 5: Download                From manifest, no re-querying needed
```

---

## Step 1: Search ENCODE for Available Data

Search all biosamples that have DNase-seq unfiltered BAMs on GRCh38, and annotate whether they also have H3K27ac and RNA-seq:

```bash
# This was done once; results saved as reports
# 369 biosamples with DNase, 109 also have H3K27ac, 206 have RNA-seq, 94 have all three
```

**Reports generated:**

| File | Description |
|------|-------------|
| [`data/encode_all_dnase_samples.html`](../data/encode_all_dnase_samples.html) | All 369 ENCODE biosamples with DNase, annotated with H3K27ac/RNA-seq availability. Searchable table. |
| [`data/encode_all_dnase_samples.tsv`](../data/encode_all_dnase_samples.tsv) | Same data as TSV (biosample, classification, organ, file counts, best accession, H3K27ac/RNA-seq flags) |

**How the search works** (in `scripts/download_encode_data.py`):

The script queries the ENCODE REST API with `frame=embedded` to get full metadata including award info (ENCODE phase) in a single request:

```
GET https://www.encodeproject.org/search/?type=File
    &status=released&assembly=GRCh38
    &biosample_ontology.term_name={name}
    &assay_title=DNase-seq
    &file_format=bam&output_type=unfiltered+alignments
    &frame=embedded&format=json
```

Key fields extracted: `accession`, `file_size`, `biological_replicates`, `date_created`, `analysis_step_version` (pipeline version), `award.rfa` (ENCODE phase 2/3/4).

---

## Step 2: Match ENCODE Biosamples to Roadmap Epigenomes

Roadmap uses its own naming (e.g., "NHEK", "HSMM"), while ENCODE uses ontology terms (e.g., "keratinocyte", "skeletal muscle myoblast"). The mapping is defined in `scripts/download_encode_data.py` → `ROADMAP_BIOSAMPLE` dict and `scripts/build_roadmap_manifest.py` → `ROADMAP_SAMPLES` dict.

**Mapping challenges:**

| Issue | Example | Resolution |
|-------|---------|------------|
| Different naming | NHEK → `keratinocyte` | Manual ontology lookup |
| Broad fallback needed | E070 Brain Germinal Matrix → `brain` | Exact term `germinal matrix` has 0 DNase BAMs |
| Source mismatch | E005 H1-BMP4 trophoblast → `trophoblast cell` | ENCODE trophoblasts are fetal, not H1-derived |
| ESC line differs | E007 H1 neural prog → `neural progenitor cell` | ENCODE has H9-derived, not H1 |
| Wrong mapping caught | E024 ES-UCSF4 was mapped to CD4+ Treg | Fixed during review — ESC, not T cell |
| No ENCODE data | E016 HUES64 | Has ENCODE files but 0 DNase BAMs |

**Reports generated:**

| File | Description |
|------|-------------|
| [`data/roadmap_encode_mapping.html`](../data/roadmap_encode_mapping.html) | Verified mapping of all 57 Roadmap epigenomes to ENCODE biosample terms. Shows match quality (exact/close/broad/none), issues, and notes. |

The Roadmap metadata file (`data/roadmap_expression/Roadmap.metadata.qc.jul2013.xlsx`) was used to get standardized names, anatomy, and cell type for each EID.

---

## Step 3: Review the Mapping

Cross-reference the mapping with actual data availability:

| File | Description |
|------|-------------|
| [`data/roadmap_encode_crossref.html`](../data/roadmap_encode_crossref.html) | All 57 Roadmap epigenomes with DNase/H3K27ac/RNA-seq availability status and completeness badges. |
| [`data/roadmap_encode_crossref.json`](../data/roadmap_encode_crossref.json) | JSON data behind the crossref report. |

**Result: 38 epigenomes have exact or close ENCODE matches with DNase data.**

| Match quality | Count | Description |
|---------------|-------|-------------|
| **Exact** | 27 | Cell line or specific cell type matches directly |
| **Close** | 11 | Similar but not identical (e.g., HUVEC → endothelial cell of umbilical vein) |
| Broad | 10 | Specific term has no DNase data; using broader category |
| None | 9 | No usable ENCODE data |
| Wrong (fixed) | 1 | E024 was incorrectly mapped |

---

## Step 4: Build the Download Manifest

For the 38 exact/close epigenomes, select the best ENCODE experiment per assay and include all biological replicates:

```bash
# Build manifest (queries ENCODE API — takes ~5 min)
python scripts/build_roadmap_manifest.py

# Or dry-run to preview without saving
python scripts/build_roadmap_manifest.py --dry-run
```

**Selection criteria:**

1. **Best experiment**: Newest `date_created` (= latest ENCODE pipeline version), then most biological replicates
2. **Best replicate** (`recommended=true`): Largest `file_size` within the experiment (= deepest sequencing depth)
3. **All replicates included**: Every biological replicate from the best experiment is in the manifest

**Reports generated:**

| File | Description |
|------|-------------|
| [`data/roadmap_download_manifest.html`](../data/roadmap_download_manifest.html) | Visual summary: 38 epigenomes × 3 assays, all replicates with clickable ENCODE/4DN links, `best` badges, per-assay summary cards. |
| [`data/roadmap_download_manifest.tsv`](../data/roadmap_download_manifest.tsv) | Tab-separated manifest (147 rows). Columns: eid, roadmap_name, encode_biosample, match_quality, assay, accession, url, file_size, bio_rep, experiment, encode_phase, pipeline_version, date_created, recommended, dest_path, source. |
| [`data/roadmap_download_manifest.json`](../data/roadmap_download_manifest.json) | JSON manifest (same data, for programmatic use). |

**Summary:**

- 125 files found (~1.1 TB), 22 not found (mostly Hi-C)
- 48 DNase, 53 H3K27ac, 24 Hi-C (+ 6 from 4DN)
- 92 files marked as recommended (best per assay per epigenome)

---

## Step 5: Download

### From manifest (recommended — no API queries)

```bash
# Download all files in the manifest
python scripts/download_encode_data.py --from-manifest data/roadmap_download_manifest.json

# Download only recommended files (best replicate per assay)
# Filter the manifest first:
python3 -c "
import json
items = [it for it in json.load(open('data/roadmap_download_manifest.json')) if it['recommended']]
json.dump(items, open('data/roadmap_download_manifest_best.json', 'w'), indent=2)
print(f'{len(items)} recommended files')
"
python scripts/download_encode_data.py --from-manifest data/roadmap_download_manifest_best.json
```

### Direct download (queries ENCODE API each time)

```bash
# All 11 cell lines (from data/cell_line_list.txt)
python scripts/download_encode_data.py

# Specific cell types
python scripts/download_encode_data.py --cell-types K562,GM12878,HepG2

# All replicates
python scripts/download_encode_data.py --all-replicates

# Save a new manifest while downloading
python scripts/download_encode_data.py --save-manifest data/my_manifest
```

### Wget/curl from TSV

The TSV manifest can be used directly with wget:

```bash
# Download all recommended files
awk -F'\t' 'NR>1 && $15=="true" && $6!="NOT_FOUND" {print $7, "-O", $16}' \
    data/roadmap_download_manifest.tsv | xargs -L1 wget -c
```

### Output directory structure

```
data/
  E003_H1/
    DNase/
      ENCFF031VZX.bam          # replicate 1 (best)
      ENCFF966UXQ.bam          # replicate 2
      ENCFF031VZX_metadata.json
    H3K27ac/
      ENCFF104RJG.bam
      ENCFF120QMN.bam
    HiC/
      ...
  E118_HepG2/
    DNase/
      ENCFF878RGP.bam
    H3K27ac/
      ENCFF490KFG.bam
    HiC/
      ENCFF805ALH.hic
  ...
```

---

## File Index

All generated data files and reports:

| File | Type | Description |
|------|------|-------------|
| **Search results** | | |
| `data/encode_all_dnase_samples.html` | Report | 369 ENCODE biosamples with DNase + H3K27ac/RNA-seq flags |
| `data/encode_all_dnase_samples.{tsv,json}` | Data | Same as above, machine-readable |
| **Mapping** | | |
| `data/roadmap_encode_mapping.html` | Report | Verified Roadmap → ENCODE biosample mapping (57 epigenomes) |
| `data/roadmap_encode_crossref.html` | Report | Crossref with data availability status |
| **Manifests** | | |
| `data/roadmap_download_manifest.html` | Report | Visual summary of download plan (38 epigenomes × 3 assays, including 4DN Hi-C matches) |
| `data/roadmap_download_manifest.{tsv,json}` | Data | Download manifest with all replicates (147 rows, ~1.1 TB, public 4DN URLs included where matched) |
| `data/encode_manifest.{tsv,json}` | Data | 11 cell lines manifest (44 rows) |
| **Scripts** | | |
| `scripts/download_encode_data.py` | Tool | ENCODE API search + download with manifest support |
| `scripts/build_roadmap_manifest.py` | Tool | Build per-replicate manifest for 38 Roadmap epigenomes with ENCODE plus systematic 4DN Hi-C matching |

---

## Regenerating Everything

If you need to refresh the data (e.g., ENCODE adds new files):

```bash
# 1. Rebuild the all-DNase-samples report (run the inline script from the conversation)
# 2. Rebuild the Roadmap manifest
python scripts/build_roadmap_manifest.py --output data/roadmap_download_manifest

# 3. Rebuild the 11 cell line manifest
python scripts/download_encode_data.py --dry-run --save-manifest data/encode_manifest \
    --report data/encode_download_report.html

# 4. Rebuild the Roadmap report
python scripts/download_encode_data.py \
    --roadmap data/roadmap_expression/.cache/EG.name.txt \
    --dry-run --report data/roadmap_encode_report.html
```
