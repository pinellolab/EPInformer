"""Promoter + enhancer sequence and signal extraction (ENSID / ABC merged tables)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyBigWig
import kipoiseq
from Bio.Seq import Seq

from .encode import one_hot_encode


def extract_promoter_enhancer_loci(
    gene_enhancer_df,
    fasta_extractor,
    signal_file=None,
    max_n_enhancer=50,
    max_distanceToTSS=150_000,
    max_seq_len=2000,
    add_flanking=True,
):
    gene_pe = gene_enhancer_df.sort_values(by="distance")
    row_0 = gene_pe.iloc[0]
    ensid = row_0["ENSID"]
    gene_tss = row_0["TSS_y"]
    chrom = "chr" + str(row_0["chrom_y"])
    strand = "+"  # row_0['strand_y']
    target_interval = kipoiseq.Interval(
        chrom, int(gene_tss - max_seq_len / 2), int(gene_tss + max_seq_len / 2)
    )
    promoter_seq = Seq(fasta_extractor.extract(target_interval))
    if strand == "-":
        promoter_seq = promoter_seq.reverse_complement()
    promoter_code = one_hot_encode(str(promoter_seq))
    enhancers_code = np.zeros((max_n_enhancer, max_seq_len, 4))
    enhancer_activity = np.zeros(max_n_enhancer)
    DNase_activity = np.zeros(max_n_enhancer)
    enhancer_distance = np.zeros(max_n_enhancer)
    enhancer_contact = np.zeros(max_n_enhancer)

    if signal_file is not None:
        signal = pyBigWig.open(signal_file, "r")
        promoter_signal = signal.values(
            chrom,
            int(gene_tss - max_seq_len / 2),
            int(gene_tss + max_seq_len / 2),
            numpy=True,
        )
        if strand == "-":
            promoter_signal = promoter_signal[::-1]
        enhancers_signal = np.zeros((max_n_enhancer, max_seq_len))
    else:
        promoter_signal = np.zeros(max_seq_len)
        enhancers_signal = np.zeros((max_n_enhancer, max_seq_len))

    gene_pe = gene_pe[
        (gene_pe["distance"] > max_seq_len / 2)
        & (gene_pe["distance"] <= max_distanceToTSS)
    ]
    e_i = 0
    ensid_element_pair = []
    for idx, row in gene_pe.iterrows():
        if pd.isna(row["start_element"]):
            continue
        if e_i >= max_n_enhancer:
            break
        enhancer_start = int(row["start_element"])
        enhancer_end = int(row["end_element"])
        enhancer_center = int((row["start_element"] + row["end_element"]) / 2)
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
            if signal_file is not None:
                enhancers_signal[e_i] = signal.values(
                    chrom,
                    enhancer_center - int(max_seq_len / 2),
                    enhancer_center + int(max_seq_len / 2),
                    numpy=True,
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
                if signal_file is not None:
                    enhancers_signal[e_i] = signal.values(
                        chrom,
                        enhancer_center - int(max_seq_len / 2),
                        enhancer_center + int(max_seq_len / 2),
                        numpy=True,
                    )
            else:
                enhancer_target_interval = kipoiseq.Interval(
                    chrom, enhancer_start, enhancer_end
                )
                enhancers_code[e_i][:enhancer_len] = one_hot_encode(
                    fasta_extractor.extract(enhancer_target_interval)
                )
                if signal_file is not None:
                    enhancers_signal[e_i][:enhancer_len] = signal.values(
                        chrom, enhancer_start, enhancer_end, numpy=True
                    )

        enhancer_activity[e_i] = row["activity_base"]
        DNase_activity[e_i] = row["normalized_dhs"]
        enhancer_distance[e_i] = row["distance"]
        enhancer_contact[e_i] = row["hic_contact"]
        ensid_element_pair.append([ensid, row["name"]])
        e_i += 1
    pe_code = np.concatenate([promoter_code[np.newaxis, :], enhancers_code], axis=0)
    pe_signal = np.concatenate([promoter_signal[np.newaxis, :], enhancers_signal])
    return (
        pe_code,
        enhancer_activity,
        DNase_activity,
        enhancer_distance,
        enhancer_contact,
        ensid,
        ensid_element_pair,
        pe_signal,
    )


def extract_promoter_enhancer_loci_signals(
    gene_enhancer_df,
    fasta_extractor,
    use_strand=False,
    signal_file=None,
    max_n_enhancer=50,
    max_distanceToTSS=150_000,
    max_seq_len=2000,
    add_flanking=True,
):
    gene_pe = gene_enhancer_df.sort_values(by="distance_relative")
    if signal_file is None or len(signal_file) == 0:
        n_signal = 0
    else:
        n_signal = len(signal_file)
    row_0 = gene_pe.iloc[0]
    ensid = row_0["ENSID"]
    gene_tss = row_0["TSS_y"]
    chrom = "chr" + str(row_0["chrom_y"])
    if use_strand:
        strand = row_0["strand_y"]
    else:
        strand = "+"
    target_interval = kipoiseq.Interval(
        chrom, int(gene_tss - max_seq_len / 2), int(gene_tss + max_seq_len / 2)
    )

    promoter_seq = Seq(fasta_extractor.extract(target_interval))
    if strand == "-":
        promoter_seq = promoter_seq.reverse_complement()
    promoter_code = one_hot_encode(str(promoter_seq))
    promoter_signals = np.zeros((max_seq_len, n_signal))
    enhancers_code = np.zeros((max_n_enhancer, max_seq_len, 4))
    enhancers_signal = np.zeros((max_n_enhancer, max_seq_len, n_signal))
    enhancer_activity = np.zeros(max_n_enhancer)
    dhs_activity = np.zeros(max_n_enhancer)
    enhancer_distance = np.zeros(max_n_enhancer)
    enhancer_contact = np.zeros(max_n_enhancer)

    signal_objs = []
    if signal_file is not None:
        for sf in signal_file:
            signal = pyBigWig.open(sf, "r")
            signal_objs.append(signal)
    if signal_file is not None:
        for s_i, signal in enumerate(signal_objs):
            promoter_signal = signal.values(
                chrom,
                int(gene_tss - max_seq_len / 2),
                int(gene_tss + max_seq_len / 2),
                numpy=True,
            )
            if strand == "-":
                promoter_signals[:, s_i] = promoter_signal[::-1]
            else:
                promoter_signals[:, s_i] = promoter_signal
    gene_pe = gene_pe[
        (gene_pe["distance"] > max_seq_len / 2)
        & (gene_pe["distance"] <= max_distanceToTSS)
    ]
    e_i = 0
    ensid_element_pair = []
    for idx, row in gene_pe.iterrows():
        if pd.isna(row["start_element"]):
            continue
        if e_i >= max_n_enhancer:
            break
        enhancer_start = int(row["start_element"])
        enhancer_end = int(row["end_element"])
        enhancer_center = int((row["start_element"] + row["end_element"]) / 2)
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
            if signal_file is not None:
                for s_i, signal in enumerate(signal_objs):
                    sv = signal.values(
                        chrom,
                        enhancer_center - int(max_seq_len / 2),
                        enhancer_center + int(max_seq_len / 2),
                        numpy=True,
                    )
                    sv = np.nan_to_num(sv)
                    enhancers_signal[e_i, :, s_i] = sv
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
                if signal_file is not None:
                    for s_i, signal in enumerate(signal_objs):
                        sv = signal.values(
                            chrom,
                            enhancer_center - int(max_seq_len / 2),
                            enhancer_center + int(max_seq_len / 2),
                            numpy=True,
                        )
                        sv = np.nan_to_num(sv)
                        enhancers_signal[e_i, :, s_i] = sv
            else:
                code_start = int(max_seq_len / 2) - int(enhancer_len / 2)
                enhancer_target_interval = kipoiseq.Interval(
                    chrom, enhancer_start, enhancer_end
                )
                enhancers_code[e_i][code_start : code_start + enhancer_len] = (
                    one_hot_encode(fasta_extractor.extract(enhancer_target_interval))
                )
                if signal_file is not None:
                    for s_i, signal in enumerate(signal_objs):
                        sv = signal.values(
                            chrom, enhancer_start, enhancer_end, numpy=True
                        )
                        sv = np.nan_to_num(sv)
                        enhancers_signal[
                            e_i, code_start : code_start + enhancer_len, s_i
                        ] = sv
        enhancer_activity[e_i] = row["activity_base"]
        dhs_activity[e_i] = row["normalized_dhs"]
        enhancer_distance[e_i] = row["distance_relative"]
        enhancer_contact[e_i] = row["hic_contact"]
        ensid_element_pair.append([ensid, row["name"]])
        e_i += 1
    pe_code = np.concatenate([promoter_code[np.newaxis, :], enhancers_code], axis=0)
    if n_signal == 0:
        pe_signal = None
    else:
        pe_signal = np.concatenate([promoter_signals[np.newaxis, :], enhancers_signal])
    return (
        pe_code,
        enhancer_activity,
        dhs_activity,
        enhancer_distance,
        enhancer_contact,
        ensid,
        ensid_element_pair,
        pe_signal,
    )
