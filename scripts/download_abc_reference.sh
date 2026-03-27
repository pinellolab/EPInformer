#!/usr/bin/env bash
# Download reference data for the ABC pipeline.
#
# Downloads gene annotations, chromosome sizes, and quantile normalization
# reference from the Broad ABC repo, plus RNA-seq RPKM from Roadmap.
#
# Usage:
#   sh scripts/download_abc_reference.sh [output_dir]
#
# Default output: data/reference/hg38/

set -euo pipefail

OUTDIR="${1:-data/reference/hg38}"
mkdir -p "$OUTDIR"

ABC_BASE="https://raw.githubusercontent.com/broadinstitute/ABC-Enhancer-Gene-Prediction/master/reference"
ROADMAP_BASE="https://egg2.wustl.edu/roadmap/data/byDataType/rna/expression"

echo "========================================"
echo "Downloading ABC reference data"
echo "Output: $OUTDIR"
echo "========================================"

# ---------- Gene annotations ----------
echo "[1/5] CollapsedGeneBounds.hg38.bed ..."
if [ ! -f "$OUTDIR/CollapsedGeneBounds.hg38.bed" ]; then
    curl -fSL "$ABC_BASE/CollapsedGeneBounds.hg38.bed" -o "$OUTDIR/CollapsedGeneBounds.hg38.bed"
else
    echo "  (already exists, skipping)"
fi

echo "[2/5] CollapsedGeneBounds.hg38.TSS500bp.bed ..."
if [ ! -f "$OUTDIR/CollapsedGeneBounds.hg38.TSS500bp.bed" ]; then
    curl -fSL "$ABC_BASE/CollapsedGeneBounds.hg38.TSS500bp.bed" -o "$OUTDIR/CollapsedGeneBounds.hg38.TSS500bp.bed"
else
    echo "  (already exists, skipping)"
fi

# ---------- Chromosome sizes ----------
echo "[3/5] GRCh38_EBV.chrom.sizes.tsv ..."
if [ ! -f "$OUTDIR/GRCh38_EBV.chrom.sizes.tsv" ]; then
    curl -fSL "$ABC_BASE/GRCh38_EBV.chrom.sizes.tsv" -o "$OUTDIR/GRCh38_EBV.chrom.sizes.tsv"
else
    echo "  (already exists, skipping)"
fi

# ---------- Quantile normalization reference ----------
echo "[4/5] EnhancersQNormRef.K562.txt ..."
if [ ! -f "$OUTDIR/EnhancersQNormRef.K562.txt" ]; then
    curl -fSL "$ABC_BASE/EnhancersQNormRef.K562.txt" -o "$OUTDIR/EnhancersQNormRef.K562.txt"
else
    echo "  (already exists, skipping)"
fi

# ---------- Roadmap RNA-seq expression ----------
echo "[5/5] Roadmap RNA-seq RPKM (57epigenomes) ..."
if [ ! -f "$OUTDIR/57epigenomes.RPKM.pc.gz" ]; then
    curl -fSL "$ROADMAP_BASE/57epigenomes.RPKM.pc.gz" -o "$OUTDIR/57epigenomes.RPKM.pc.gz"
else
    echo "  (already exists, skipping)"
fi

echo ""
echo "========================================"
echo "Download complete!"
echo "Files in: $OUTDIR"
echo "========================================"
ls -lh "$OUTDIR"
