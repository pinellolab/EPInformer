"""
ABC-style gene–enhancer tables: encode promoter + candidate enhancers as one-hot arrays.

Expected columns (typical ABC EnhancerPredictions export): ``TargetGene``, ``TargetGeneTSS``,
``chr``, ``distance``, ``start``, ``end``, ``name``, ``activity_base``, ``hic_contact``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import kipoiseq

from epinformer_preprocessing.encode import one_hot_encode
from epinformer_preprocessing.fasta import FastaStringExtractor


def encode_promoter_enhancer_links(
    gene_enhancer_df,
    fasta_path: str = "./data/hg38.fa",
    max_n_enhancer: int = 60,
    max_distanceToTSS: int = 100_000,
    max_seq_len: int = 2000,
    add_flanking: bool = False,
):
    """
    Encode one gene's rows (sorted by distance) to promoter + padded enhancer sequences.

    Returns ``None`` tuples if ``gene_enhancer_df`` is empty (same as legacy predict script).
    """
    fasta_extractor = FastaStringExtractor(fasta_path)
    gene_pe = gene_enhancer_df.sort_values(by="distance")
    if gene_pe.empty:
        print("Warning: gene_pe is empty. Skipping encoding.")
        return None, None, None, None, None, None
    row_0 = gene_pe.iloc[0]
    gene_name = row_0["TargetGene"]
    gene_tss = row_0["TargetGeneTSS"]
    chrom = row_0["chr"]
    if row_0["TargetGeneTSS"] != row_0["TargetGeneTSS"]:
        gene_tss = row_0["tss"]
        gene_name = row_0["name_gene"]
        chrom = row_0["chr"]

    target_interval = kipoiseq.Interval(
        chrom, int(gene_tss - max_seq_len / 2), int(gene_tss + max_seq_len / 2)
    )
    promoter_seq = fasta_extractor.extract(target_interval)
    promoter_code = one_hot_encode(promoter_seq)
    enhancers_code = np.zeros((max_n_enhancer, max_seq_len, 4))
    enhancer_activity = np.zeros(max_n_enhancer)
    enhancer_distance = np.zeros(max_n_enhancer)
    enhancer_contact = np.zeros(max_n_enhancer)
    gene_pe = gene_pe[
        (gene_pe["distance"] > max_seq_len / 2)
        & (gene_pe["distance"] <= max_distanceToTSS)
    ]
    e_i = 0
    gene_element_pair = []
    for idx, row in gene_pe.iterrows():
        if row["TargetGene"] != row["TargetGene"]:
            break
        if pd.isna(row["start"]):
            continue
        if e_i >= max_n_enhancer:
            break
        enhancer_start = int(row["start"])
        enhancer_end = int(row["end"])
        enhancer_center = int((row["start"] + row["end"]) / 2)
        enhancer_len = enhancer_end - enhancer_start
        if add_flanking:
            enhancer_target_interval = kipoiseq.Interval(
                chrom,
                enhancer_center - int(max_seq_len / 2),
                enhancer_center + int(max_seq_len / 2),
            )
            enhancers_code[e_i][:] = one_hot_encode(
                fasta_extractor.extract(enhancer_target_interval)
            )
        else:
            if enhancer_len > max_seq_len:
                enhancer_target_interval = kipoiseq.Interval(
                    chrom,
                    enhancer_center - int(max_seq_len / 2),
                    enhancer_center + int(max_seq_len / 2),
                )
                enhancers_code[e_i][:] = one_hot_encode(
                    fasta_extractor.extract(enhancer_target_interval)
                )
            else:
                code_start = int(max_seq_len / 2) - int(enhancer_len / 2)
                enhancer_target_interval = kipoiseq.Interval(
                    chrom, enhancer_start, enhancer_end
                )
                enhancers_code[e_i][code_start : code_start + enhancer_len] = (
                    one_hot_encode(fasta_extractor.extract(enhancer_target_interval))
                )
        enhancer_activity[e_i] = row["activity_base"]
        enhancer_distance[e_i] = row["distance"]
        enhancer_contact[e_i] = row["hic_contact"]
        gene_element_pair.append([gene_name, row["name"]])
        e_i += 1
    pe_code = np.concatenate([promoter_code[np.newaxis, :], enhancers_code], axis=0)
    gene_element_pair = pd.DataFrame(gene_element_pair, columns=["gene", "element"])
    return (
        pe_code,
        enhancer_activity,
        enhancer_distance,
        enhancer_contact,
        gene_name,
        gene_element_pair,
    )
