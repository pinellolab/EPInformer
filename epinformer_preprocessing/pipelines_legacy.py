"""
Legacy one-off preprocessing drivers (hardcoded paths) and :class:`DataGenerator_seqs2Expr`.

Outputs **compressed HDF5** ``samples.h5`` per run (see :mod:`epinformer_preprocessing.hdf5`),
plus ``gene_enhancer_pair.csv`` sidecars. Per-gene ``.npy`` files are no longer written.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import tqdm

from epinformer_preprocessing.extract import (
    extract_promoter_enhancer_loci,
    extract_promoter_enhancer_loci_signals,
)
from epinformer_preprocessing.fasta import FastaStringExtractor
from epinformer_preprocessing.hdf5 import (
    create_pe_arrays_h5,
    write_enhancer,
    write_gene_sample,
)
from epinformer_preprocessing.encode import one_hot_encode

_DEFAULT_GENE_EXPR_CSV = "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/GM12878_K562_18377_gene_expression.csv"
_DEFAULT_FASTA_PINELLO = (
    "/data/pinello/PROJECTS/2020_06_Pred_Gene_expression/EPInformerV2_20230209/hg38.fa"
)
_DEFAULT_K562_H3_PREDICTIONS = (
    "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/"
    "DNase_ENCFF257HEE_overlapH3K27acENCFF544LXB_noCutOff_hic_1MB_predictions//"
    "EnhancerPredictionsAllPutative.txt.gz"
)
_DEFAULT_K562_DNASE_BW = (
    "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/DNase/ENCFF972GVB.bigWig"
)
_DEFAULT_K562_SIGNAL_FILES = [
    "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/DNase/ENCFF972GVB.bigWig",
    "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/H3K27ac/ENCFF465GBD.bigWig",
    "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/H3K4me1/ENCFF287LBI.bigWig",
    "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/H3K4me3/ENCFF144MRB.bigWig",
    "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/CTCF/ENCFF168IFW.bigWig",
]
_DEFAULT_K562_H3_OUT = (
    "./seqs2expr_dataset/K562_DNase_ENCFF257HEE_overlapH3K27acENCFF544LXB_"
    "1kbNoFlankSeq_noCutOff_hic_150kb60e_AllPutative_signal"
)


def _apply_tss_column(cage_xpresso_df: pd.DataFrame, tss_column: str) -> pd.DataFrame:
    """Rename alternate TSS column (e.g. TSS_xpresso) to ``TSS`` for downstream code."""
    if tss_column != "TSS":
        if tss_column not in cage_xpresso_df.columns:
            raise ValueError(
                f"TSS column {tss_column!r} not found in gene expression CSV; "
                f"have: {list(cage_xpresso_df.columns)[:20]}..."
            )
        return cage_xpresso_df.rename(columns={tss_column: "TSS"})
    return cage_xpresso_df


def _orf_density_alias(cage_xpresso_df: pd.DataFrame) -> pd.DataFrame:
    """Map Xpresso-style column name to legacy ``ORFEXONDENSITY`` if needed."""
    if (
        "ORFEXONDENSITY" not in cage_xpresso_df.columns
        and "ORFEXONDENSITY_log10zscore" in cage_xpresso_df.columns
    ):
        return cage_xpresso_df.rename(
            columns={"ORFEXONDENSITY_log10zscore": "ORFEXONDENSITY"}
        )
    return cage_xpresso_df


def _map_symbol_to_ensid(promoter_enhancer_df: pd.DataFrame, cage_xpresso_df: pd.DataFrame) -> pd.DataFrame:
    """Map ABC TargetGene (gene symbol) → ENSID using expression CSV lookup.

    ABC output uses gene symbols (e.g. BET1L) while the expression CSV indexes
    by ENSID (e.g. ENSG00000000003).  This creates a ``TargetGene_ENSID`` column
    so the downstream merge can join on ENSID.
    """
    symbol_to_ensid = cage_xpresso_df.set_index("Gene name")["ENSID"].to_dict()
    promoter_enhancer_df["TargetGene_ENSID"] = promoter_enhancer_df["TargetGene"].map(symbol_to_ensid)
    n_mapped = promoter_enhancer_df["TargetGene_ENSID"].notna().sum()
    n_genes = promoter_enhancer_df.loc[promoter_enhancer_df["TargetGene_ENSID"].notna(), "TargetGene"].nunique()
    n_total_genes = promoter_enhancer_df["TargetGene"].nunique()
    print(f"Gene symbol → ENSID: {n_genes}/{n_total_genes} genes mapped ({n_mapped}/{len(promoter_enhancer_df)} rows)")
    return promoter_enhancer_df


class DataGenerator_seqs2Expr(torch.utils.data.Dataset):
    def __init__(
        self, pe_sequences, rna_feat_list, distance_list, ensid_list, cell_type="CAGE"
    ):
        self.cell_type = cell_type
        self.pe_sequences = pe_sequences
        self.distance_list = distance_list
        self.rna_feat_list = rna_feat_list
        self.ensid_list = ensid_list
        self.expr_df = pd.read_csv(
            "/data/pinello/PROJECTS/2020_06_Pred_Gene_expression/EPInformerV2_20230209/GM12878_K562_18377_gene_expression.csv",
            index_col="ENSID",
        )

    def __len__(self):
        return len(self.ensid_list)

    def __getitem__(self, idx):
        ensid = self.ensid_list[idx]
        X_seq = self.pe_sequences[idx]
        X_rnaFeat = self.rna_feat_list[idx]
        if self.expr_type == "CAGE":
            y_expr = np.log10(
                self.expr_df.loc[ensid][self.cell_type + "_CAGE_128*3_sum"] + 1
            )
        elif self.expr_type == "RNA":
            y_expr = self.expr_df.loc[ensid][self.cell_type + "_RNArpkm"]
        else:
            assert False, "label not exists!"
        return X_seq, X_rnaFeat, y_expr, ensid


def obtain_PE(
    fname,
    signal_files,
    max_distance=150_000,
    add_flank=False,
    use_strand=False,
    n_enhancer=60,
    max_seq_len=2000,
    pe_type="AllPutative",
    cell_type="K562",
    gene_expression_csv: Optional[str] = None,
    fasta_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    tss_column: str = "TSS",
):
    pe_fname = fname[0]
    enhancer_fname = fname[1]
    promoter_enhancer_df = pd.read_csv(pe_fname, sep="\t")
    promoter_enhancer_df["element_mid"] = (
        promoter_enhancer_df["end"] + promoter_enhancer_df["start"]
    ) / 2
    promoter_enhancer_df["distance_relative"] = (
        promoter_enhancer_df["element_mid"] - promoter_enhancer_df["TargetGeneTSS"]
    )

    enhancer_df = pd.read_csv(enhancer_fname, sep="\t")[
        ["name", "normalized_h3K27ac", "normalized_dhs"]
    ]
    promoter_enhancer_df = promoter_enhancer_df.merge(enhancer_df, on="name")
    print(promoter_enhancer_df.columns)
    prefix_name = pe_fname.split("/")[-2].split("_noCutOff_")[0]

    _ge = gene_expression_csv or _DEFAULT_GENE_EXPR_CSV
    cage_xpresso_df = pd.read_csv(_ge)
    cage_xpresso_df = _apply_tss_column(cage_xpresso_df, tss_column)
    cage_xpresso_df = _orf_density_alias(cage_xpresso_df)
    cage_xpresso_df = cage_xpresso_df[
        [
            "ENSID",
            "chrom",
            "start",
            "end",
            "strand",
            "TSS",
            "Gene name",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY",
        ]
    ]
    promoter_enhancer_df = _map_symbol_to_ensid(promoter_enhancer_df, cage_xpresso_df)
    gene_enhancer_expr_df = promoter_enhancer_df.merge(
        cage_xpresso_df,
        left_on="TargetGene_ENSID",
        right_on="ENSID",
        how="right",
        suffixes=["_element", "_gene"],
    ).sort_values(by=["ENSID", "distance"])
    gene_enhancer_expr_df = gene_enhancer_expr_df[
        [
            "chrom",
            "start_element",
            "end_element",
            "hic_contact",
            "activity_base",
            "normalized_h3K27ac",
            "normalized_dhs",
            "name",
            "distance",
            "distance_relative",
            "ENSID",
            "TSS",
            "strand",
            "Gene name",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY",
        ]
    ]
    for distanceToTSS in [
        50_000,
        100_000,
        120_000,
        150_000,
        250_000,
        350_000,
        400_000,
        500_000,
        600_000,
    ]:
        print(distanceToTSS)
        count_df = gene_enhancer_expr_df[
            (gene_enhancer_expr_df["distance"] <= distanceToTSS)
            & (gene_enhancer_expr_df["distance"] > max_seq_len / 2)
        ].groupby("ENSID")["name"].count()
        print(
            "90%",
            np.percentile(count_df, 90),
            "95%",
            np.percentile(count_df, 95),
            np.percentile(count_df, 100),
        )

    _fa = fasta_path or _DEFAULT_FASTA_PINELLO
    fasta_extractor = FastaStringExtractor(_fa)
    gene_enhancer_expr_sub_df = gene_enhancer_expr_df[
        (gene_enhancer_expr_df["distance"] <= max_distance)
        & (gene_enhancer_expr_df["distance"] > max_seq_len / 2)
    ]

    gene_tss_df = gene_enhancer_expr_df.drop_duplicates(subset=["ENSID"])[
        ["ENSID", "TSS", "chrom", "strand"]
    ].reset_index()
    gene_enhancer_expr_sub_df = gene_enhancer_expr_sub_df.merge(
        gene_tss_df, on="ENSID", how="right", suffixes=["_x", "_y"]
    )
    if output_dir is not None:
        out_folder = output_dir
    elif add_flank:
        out_folder = (
            "./seqs2expr_dataset/"
            + cell_type
            + "_"
            + prefix_name
            + "_"
            + str(int(max_seq_len / 1000))
            + "kb_noCutOff_relDist_FlankSeq_"
            + str(int(max_distance / 1000))
            + "kb"
            + str(n_enhancer)
            + "e_"
            + pe_type
            + "_signals_"
            + str(use_strand)
        )
    else:
        out_folder = (
            "./seqs2expr_dataset/"
            + cell_type
            + "_"
            + prefix_name
            + "_"
            + str(int(max_seq_len / 1000))
            + "kb_noCutOff_relDist_noFlankSeq_"
            + str(int(max_distance / 1000))
            + "kb"
            + str(n_enhancer)
            + "e_"
            + pe_type
            + "_signals_"
            + str(use_strand)
        )
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print(out_folder)
    ensid_pair_list = []
    ensid_list = list(set(gene_enhancer_expr_sub_df["ENSID"]))
    n_sig = len(signal_files)
    h5_path = os.path.join(out_folder, "samples.h5")
    hf = create_pe_arrays_h5(
        h5_path,
        n_samples=len(ensid_list),
        max_n_enhancer=n_enhancer,
        max_seq_len=max_seq_len,
        n_signal_tracks=n_sig,
    )
    for row_i, ensid in enumerate(
        tqdm.tqdm(ensid_list)
    ):
        gene_df = gene_enhancer_expr_sub_df[gene_enhancer_expr_sub_df["ENSID"] == ensid]
        (
            seq_code,
            act_list,
            dhs_list,
            dist_list,
            contact_list,
            _ens,
            ensid_element_pair,
            seq_signal,
        ) = extract_promoter_enhancer_loci_signals(
            gene_df,
            fasta_extractor,
            use_strand=use_strand,
            signal_file=None if n_sig == 0 else signal_files,
            max_seq_len=max_seq_len,
            max_n_enhancer=n_enhancer,
            max_distanceToTSS=max_distance,
            add_flanking=add_flank,
        )
        ensid_pair_list = ensid_pair_list + ensid_element_pair
        write_pe_sample(
            hf,
            row_i,
            str(ensid),
            seq_code,
            act_list,
            dhs_list,
            dist_list,
            contact_list,
            seq_signal,
            n_signal_tracks=n_sig,
        )
    hf.close()
    gene_enhancer_df = pd.DataFrame(ensid_pair_list, columns=["ensid", "element"])
    gene_enhancer_df.to_csv(out_folder + "/gene_enhancer_pair.csv", index=False)
    gene_enhancer_expr_sub_df.to_csv(out_folder + "/gene_enhancer_pair_all.csv", index=False)


def obtain_K562_H3K27ac_PE(
    predictions_tsv: Optional[str] = None,
    gene_expression_csv: Optional[str] = None,
    fasta_path: Optional[str] = None,
    dnase_bigwig: Optional[str] = None,
    output_dir: Optional[str] = None,
    tss_column: str = "TSS",
    include_bigwig: bool = True,
):
    _pred = predictions_tsv or _DEFAULT_K562_H3_PREDICTIONS
    promoter_enhancer_df = pd.read_csv(_pred, sep="\t")
    _ge = gene_expression_csv or _DEFAULT_GENE_EXPR_CSV
    cage_xpresso_df = pd.read_csv(_ge)
    cage_xpresso_df = _apply_tss_column(cage_xpresso_df, tss_column)
    cage_xpresso_df = cage_xpresso_df[
        [
            "ENSID",
            "chrom",
            "start",
            "end",
            "strand",
            "TSS",
            "Gene name",
            "K562_CAGE_128*3_sum",
            "Actual_K562",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY_log10zscore",
        ]
    ]
    promoter_enhancer_df = _map_symbol_to_ensid(promoter_enhancer_df, cage_xpresso_df)
    gene_enhancer_expr_df = promoter_enhancer_df.merge(
        cage_xpresso_df,
        left_on="TargetGene_ENSID",
        right_on="ENSID",
        how="right",
        suffixes=["_element", "_gene"],
    ).sort_values(by=["ENSID", "distance"])
    gene_enhancer_expr_df = gene_enhancer_expr_df[
        [
            "chrom",
            "start_element",
            "end_element",
            "hic_contact",
            "activity_base",
            "name",
            "distance",
            "ENSID",
            "TSS",
            "strand",
            "Gene name",
            "K562_CAGE_128*3_sum",
            "Actual_K562",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY_log10zscore",
        ]
    ]
    for distanceToTSS in [
        50_000,
        100_000,
        150_000,
        250_000,
        350_000,
        400_000,
        500_000,
        600_000,
    ]:
        print(distanceToTSS)
        count_df = gene_enhancer_expr_df[
            gene_enhancer_expr_df["distance"] <= distanceToTSS
        ].groupby("ENSID")["name"].count()
        print(
            "90%",
            np.percentile(count_df, 90),
            "95%",
            np.percentile(count_df, 95),
            np.percentile(count_df, 100),
        )

    _fa = fasta_path or _DEFAULT_FASTA_PINELLO
    fasta_extractor = FastaStringExtractor(_fa)
    gene_enhancer_expr_sub_df = gene_enhancer_expr_df[
        (gene_enhancer_expr_df["distance"] <= 150_000)
        & (gene_enhancer_expr_df["distance"] > 500)
    ]
    gene_tss_df = gene_enhancer_expr_df.drop_duplicates(subset=["ENSID"])[
        ["ENSID", "TSS", "chrom", "strand"]
    ].reset_index()
    gene_enhancer_expr_sub_df = gene_enhancer_expr_sub_df.merge(
        gene_tss_df, on="ENSID", how="right", suffixes=["_x", "_y"]
    )
    out_folder = output_dir or _DEFAULT_K562_H3_OUT

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print(out_folder)
    ensid_pair_list = []
    if include_bigwig:
        bw: Optional[str] = dnase_bigwig or _DEFAULT_K562_DNASE_BW
        n_bw_tracks = 1
    else:
        bw = None
        n_bw_tracks = 0
    ensid_list = list(set(gene_enhancer_expr_sub_df["ENSID"]))
    n_enh = 60
    L = 1000
    h5_path = os.path.join(out_folder, "samples.h5")
    hf = create_pe_arrays_h5(
        h5_path,
        n_samples=len(ensid_list),
        max_n_enhancer=n_enh,
        max_seq_len=L,
        n_signal_tracks=n_bw_tracks,
    )
    for row_i, ensid in enumerate(tqdm.tqdm(ensid_list)):
        gene_df = gene_enhancer_expr_sub_df[gene_enhancer_expr_sub_df["ENSID"] == ensid]
        (
            seq_code,
            act_list,
            dhs_list,
            dist_list,
            contact_list,
            _ens,
            ensid_element_pair,
            seq_signal,
        ) = extract_promoter_enhancer_loci(
            gene_df,
            fasta_extractor,
            signal_file=bw,
            max_seq_len=L,
            max_n_enhancer=n_enh,
            max_distanceToTSS=150_000,
            add_flanking=False,
        )
        ensid_pair_list = ensid_pair_list + ensid_element_pair
        write_pe_sample(
            hf,
            row_i,
            str(ensid),
            seq_code,
            act_list,
            dhs_list,
            dist_list,
            contact_list,
            seq_signal,
            n_signal_tracks=n_bw_tracks,
        )
    hf.close()
    gene_enhancer_df = pd.DataFrame(ensid_pair_list, columns=["ensid", "element"])
    gene_enhancer_df.to_csv(out_folder + "/gene_enhancer_pair.csv", index=False)
    gene_enhancer_expr_sub_df.to_csv(out_folder + "/gene_enhancer_pair_all.csv", index=False)


def obtain_GM12878_PE(max_distance=150_000, add_flank=False, n_enhancer=60, max_seq_len=2000):
    promoter_enhancer_df = pd.read_csv(
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/GM12878_data/DNase_ENCFF729UYK_hichip_noCutOff_1MB_predictions/EnhancerPredictionsAllPutative.txt.gz",
        sep="\t",
    )
    cage_xpresso_df = pd.read_csv(
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/GM12878_K562_18377_gene_expression.csv"
    )
    cage_xpresso_df = cage_xpresso_df[
        [
            "ENSID",
            "chrom",
            "start",
            "end",
            "strand",
            "TSS",
            "Gene name",
            "K562_CAGE_128*3_sum",
            "Actual_K562",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY",
        ]
    ]
    promoter_enhancer_df = _map_symbol_to_ensid(promoter_enhancer_df, cage_xpresso_df)
    gene_enhancer_expr_df = promoter_enhancer_df.merge(
        cage_xpresso_df,
        left_on="TargetGene_ENSID",
        right_on="ENSID",
        how="right",
        suffixes=["_element", "_gene"],
    ).sort_values(by=["ENSID", "distance"])
    gene_enhancer_expr_df = gene_enhancer_expr_df[
        [
            "chrom",
            "start_element",
            "end_element",
            "hic_contact",
            "activity_base",
            "name",
            "distance",
            "ENSID",
            "TSS",
            "strand",
            "Gene name",
            "K562_CAGE_128*3_sum",
            "Actual_K562",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY",
        ]
    ]
    for distanceToTSS in [
        50_000,
        100_000,
        120_000,
        150_000,
        250_000,
        350_000,
        400_000,
        500_000,
        600_000,
    ]:
        print(distanceToTSS)
        count_df = gene_enhancer_expr_df[
            (gene_enhancer_expr_df["distance"] <= distanceToTSS)
            & (gene_enhancer_expr_df["distance"] > max_seq_len / 2)
        ].groupby("ENSID")["name"].count()
        print(
            "90%",
            np.percentile(count_df, 90),
            "95%",
            np.percentile(count_df, 95),
            np.percentile(count_df, 100),
        )

    fasta_extractor = FastaStringExtractor(
        "/data/pinello/PROJECTS/2020_06_Pred_Gene_expression/EPInformerV2_20230209//hg38.fa"
    )
    gene_enhancer_expr_sub_df = gene_enhancer_expr_df[
        (gene_enhancer_expr_df["distance"] <= max_distance)
        & (gene_enhancer_expr_df["distance"] > max_seq_len / 2)
    ]
    gene_tss_df = gene_enhancer_expr_df.drop_duplicates(subset=["ENSID"])[
        ["ENSID", "TSS", "chrom", "strand"]
    ].reset_index()
    gene_enhancer_expr_sub_df = gene_enhancer_expr_sub_df.merge(
        gene_tss_df, on="ENSID", how="right", suffixes=["_x", "_y"]
    )
    out_folder = "./seqs2expr_dataset/GM12878_DNase_ENCFF729UYK_1kbNoFlankSeq_noCutOff_hic_150kb60e_AllPutative_signal"

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print(out_folder)
    ensid_pair_list = []
    bw = "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/GM12878_data/DNase/ENCFF960FMM.bigWig"
    ensid_list = list(set(gene_enhancer_expr_sub_df["ENSID"]))
    h5_path = os.path.join(out_folder, "samples.h5")
    hf = create_pe_arrays_h5(
        h5_path,
        n_samples=len(ensid_list),
        max_n_enhancer=n_enhancer,
        max_seq_len=max_seq_len,
        n_signal_tracks=1,
    )
    for row_i, ensid in enumerate(tqdm.tqdm(ensid_list)):
        gene_df = gene_enhancer_expr_sub_df[gene_enhancer_expr_sub_df["ENSID"] == ensid]
        (
            seq_code,
            act_list,
            dhs_list,
            dist_list,
            contact_list,
            _ens,
            ensid_element_pair,
            seq_signal,
        ) = extract_promoter_enhancer_loci(
            gene_df,
            fasta_extractor,
            signal_file=bw,
            max_seq_len=max_seq_len,
            max_n_enhancer=n_enhancer,
            max_distanceToTSS=max_distance,
            add_flanking=add_flank,
        )
        ensid_pair_list = ensid_pair_list + ensid_element_pair
        write_pe_sample(
            hf,
            row_i,
            str(ensid),
            seq_code,
            act_list,
            dhs_list,
            dist_list,
            contact_list,
            seq_signal,
            n_signal_tracks=1,
        )
    hf.close()
    gene_enhancer_df = pd.DataFrame(ensid_pair_list, columns=["ensid", "element"])
    gene_enhancer_df.to_csv(out_folder + "/gene_enhancer_pair.csv", index=False)
    gene_enhancer_expr_sub_df.to_csv(out_folder + "/gene_enhancer_pair_all.csv", index=False)


def obtain_GM12878_H3K27ac_PE():
    promoter_enhancer_df = pd.read_csv(
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/GM12878_data/DNase_ENCFF729UYK_overlapH3K27acENCFF023LTU_hichip_noCutOff_1MB_predictions//EnhancerPredictionsAllPutative.txt.gz",
        sep="\t",
    )
    cage_xpresso_df = pd.read_csv(
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/GM12878_K562_18377_gene_expression.csv"
    )
    cage_xpresso_df = cage_xpresso_df[
        [
            "ENSID",
            "chrom",
            "start",
            "end",
            "strand",
            "TSS",
            "Gene name",
            "K562_CAGE_128*3_sum",
            "Actual_K562",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY_log10zscore",
        ]
    ]
    promoter_enhancer_df = _map_symbol_to_ensid(promoter_enhancer_df, cage_xpresso_df)
    gene_enhancer_expr_df = promoter_enhancer_df.merge(
        cage_xpresso_df,
        left_on="TargetGene_ENSID",
        right_on="ENSID",
        how="right",
        suffixes=["_element", "_gene"],
    ).sort_values(by=["ENSID", "distance"])
    gene_enhancer_expr_df = gene_enhancer_expr_df[
        [
            "chrom",
            "start_element",
            "end_element",
            "hic_contact",
            "activity_base",
            "name",
            "distance",
            "ENSID",
            "TSS",
            "strand",
            "Gene name",
            "K562_CAGE_128*3_sum",
            "Actual_K562",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY_log10zscore",
        ]
    ]
    for distanceToTSS in [
        50_000,
        100_000,
        150_000,
        250_000,
        350_000,
        400_000,
        500_000,
        600_000,
    ]:
        print(distanceToTSS)
        count_df = gene_enhancer_expr_df[
            gene_enhancer_expr_df["distance"] <= distanceToTSS
        ].groupby("ENSID")["name"].count()
        print(
            "90%",
            np.percentile(count_df, 90),
            "95%",
            np.percentile(count_df, 95),
            np.percentile(count_df, 100),
        )

    fasta_extractor = FastaStringExtractor(
        "/data/pinello/PROJECTS/2020_06_Pred_Gene_expression/EPInformerV2_20230209//hg38.fa"
    )
    gene_enhancer_expr_sub_df = gene_enhancer_expr_df[
        (gene_enhancer_expr_df["distance"] <= 150_000)
        & (gene_enhancer_expr_df["distance"] > 500)
    ]
    gene_tss_df = gene_enhancer_expr_df.drop_duplicates(subset=["ENSID"])[
        ["ENSID", "TSS", "chrom", "strand"]
    ].reset_index()
    gene_enhancer_expr_sub_df = gene_enhancer_expr_sub_df.merge(
        gene_tss_df, on="ENSID", how="right", suffixes=["_x", "_y"]
    )
    out_folder = "./seqs2expr_dataset/GM12878_DNase_ENCFF729UYK_overlapH3K27acENCFF023LTU_1kbNoFlankSeq_noCutOff_hichip_150kb60e_AllPutative_signal"

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print(out_folder)
    ensid_pair_list = []
    bw = "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/GM12878_data/DNase/ENCFF960FMM.bigWig"
    ensid_list = list(set(gene_enhancer_expr_sub_df["ENSID"]))
    n_enh = 60
    L = 1000
    h5_path = os.path.join(out_folder, "samples.h5")
    hf = create_pe_arrays_h5(
        h5_path,
        n_samples=len(ensid_list),
        max_n_enhancer=n_enh,
        max_seq_len=L,
        n_signal_tracks=1,
    )
    for row_i, ensid in enumerate(tqdm.tqdm(ensid_list)):
        gene_df = gene_enhancer_expr_sub_df[gene_enhancer_expr_sub_df["ENSID"] == ensid]
        (
            seq_code,
            act_list,
            dhs_list,
            dist_list,
            contact_list,
            _ens,
            ensid_element_pair,
            seq_signal,
        ) = extract_promoter_enhancer_loci(
            gene_df,
            fasta_extractor,
            signal_file=bw,
            max_seq_len=L,
            max_n_enhancer=n_enh,
            max_distanceToTSS=150_000,
            add_flanking=False,
        )
        ensid_pair_list = ensid_pair_list + ensid_element_pair
        write_pe_sample(
            hf,
            row_i,
            str(ensid),
            seq_code,
            act_list,
            dhs_list,
            dist_list,
            contact_list,
            seq_signal,
            n_signal_tracks=1,
        )
    hf.close()
    gene_enhancer_df = pd.DataFrame(ensid_pair_list, columns=["ensid", "element"])
    gene_enhancer_df.to_csv(out_folder + "/gene_enhancer_pair.csv", index=False)
    gene_enhancer_expr_sub_df.to_csv(out_folder + "/gene_enhancer_pair_all.csv", index=False)


def obtain_PE_withSignals(
    fname,
    max_distance=150_000,
    min_distance=0,
    add_flank=False,
    n_enhancer=60,
    max_seq_len=2000,
    gene_expression_csv: Optional[str] = None,
    fasta_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    signal_files: Optional[list[str]] = None,
    tss_column: str = "TSS",
    include_self_promoter: bool = False,
    abc_all_putative: Optional[str] = None,
    cell_type: str = "K562",
):
    """General version of obtain_K562_PE_withSignals that works for any cell type.

    Matches ABC ``TargetGene`` (ENSID) with ``Actual_{cell_type}`` expression column.
    """
    pe_fname = fname[0]
    enhancer_fname = fname[1]
    promoter_enhancer_df = pd.read_csv(pe_fname, sep="\t")
    promoter_enhancer_df["element_mid"] = (
        promoter_enhancer_df["end"] + promoter_enhancer_df["start"]
    ) / 2
    promoter_enhancer_df["distance_relative"] = (
        promoter_enhancer_df["element_mid"] - promoter_enhancer_df["TargetGeneTSS"]
    )
    enhancer_df = pd.read_csv(enhancer_fname, sep="\t")[
        ["name", "normalized_h3K27ac", "normalized_dhs"]
    ]
    promoter_enhancer_df = promoter_enhancer_df.merge(enhancer_df, on="name", suffixes=["_pred", "_enh"])
    for col in ["normalized_h3K27ac", "normalized_dhs"]:
        if col + "_enh" in promoter_enhancer_df.columns:
            promoter_enhancer_df[col] = promoter_enhancer_df[col + "_enh"]
            promoter_enhancer_df.drop(columns=[col + "_pred", col + "_enh"], inplace=True)

    # --- Load gene expression and resolve cell-type column ---
    _ge = gene_expression_csv or _DEFAULT_GENE_EXPR_CSV
    cage_xpresso_df = pd.read_csv(_ge)
    cage_xpresso_df = _apply_tss_column(cage_xpresso_df, tss_column)
    cage_xpresso_df = _orf_density_alias(cage_xpresso_df)

    actual_col = f"{cell_type}_RNArpkm"
    if actual_col not in cage_xpresso_df.columns:
        # Fallback: try legacy Actual_{cell_type} naming
        legacy_col = f"Actual_{cell_type}"
        if legacy_col in cage_xpresso_df.columns:
            actual_col = legacy_col
        else:
            raise ValueError(
                f"Expression column '{actual_col}' (or '{legacy_col}') not found in {_ge}. "
                f"Available expression columns: "
                f"{[c for c in cage_xpresso_df.columns if c.endswith('_RNArpkm') or c.startswith('Actual_')]}"
            )

    base_cols = [
        "ENSID", "chrom", "start", "end", "strand", "TSS", "Gene name",
        actual_col,
        "UTR5LEN_log10zscore", "CDSLEN_log10zscore", "INTRONLEN_log10zscore",
        "UTR3LEN_log10zscore", "UTR5GC", "CDSGC", "UTR3GC", "ORFEXONDENSITY",
    ]
    cage_xpresso_df = cage_xpresso_df[base_cols]

    promoter_enhancer_df = _map_symbol_to_ensid(promoter_enhancer_df, cage_xpresso_df)

    gene_enhancer_expr_df = promoter_enhancer_df.merge(
        cage_xpresso_df,
        left_on="TargetGene_ENSID",
        right_on="ENSID",
        how="right",
        suffixes=["_element", "_gene"],
    ).sort_values(by=["ENSID", "distance"])

    merge_cols = [
        "chrom", "start_element", "end_element", "hic_contact", "activity_base",
        "normalized_dhs", "name", "distance", "distance_relative",
        "ENSID", "TSS", "strand", "Gene name",
        actual_col,
        "UTR5LEN_log10zscore", "CDSLEN_log10zscore", "INTRONLEN_log10zscore",
        "UTR3LEN_log10zscore", "UTR5GC", "CDSGC", "UTR3GC", "ORFEXONDENSITY",
    ]
    gene_enhancer_expr_df = gene_enhancer_expr_df[merge_cols]

    # Report gene matching
    n_abc_genes = promoter_enhancer_df["TargetGene"].nunique()
    n_matched = gene_enhancer_expr_df["ENSID"].nunique()
    print(f"Gene matching: {n_matched}/{n_abc_genes} ABC genes found in expression CSV ({actual_col})")

    for distanceToTSS in [50_000, 100_000, 150_000, 250_000, 500_000]:
        count_df = gene_enhancer_expr_df[
            (gene_enhancer_expr_df["distance"] <= distanceToTSS)
            & (gene_enhancer_expr_df["distance"] > max_seq_len / 2)
        ].groupby("ENSID")["name"].count()
        if len(count_df) > 0:
            print(f"  {distanceToTSS/1000:.0f}kb: 90%={np.percentile(count_df, 90):.0f}, "
                  f"95%={np.percentile(count_df, 95):.0f}, max={np.percentile(count_df, 100):.0f}")

    _fa = fasta_path or _DEFAULT_FASTA_PINELLO
    fasta_extractor = FastaStringExtractor(_fa)
    _min_dist = max(min_distance, max_seq_len / 2)
    gene_enhancer_expr_sub_df = gene_enhancer_expr_df[
        (gene_enhancer_expr_df["distance"] <= max_distance)
        & (gene_enhancer_expr_df["distance"] > _min_dist)
    ]
    gene_tss_df = gene_enhancer_expr_df.drop_duplicates(subset=["ENSID"])[
        ["ENSID", "TSS", "chrom", "strand"]
    ].reset_index()
    gene_enhancer_expr_sub_df = gene_enhancer_expr_sub_df.merge(
        gene_tss_df, on="ENSID", how="right", suffixes=["_x", "_y"]
    )

    # ── Self-promoter lookup ───────────────────────────────────────────
    self_prm_lookup = {}
    if include_self_promoter:
        _abc_path = abc_all_putative or pe_fname
        print(f"Loading self-promoter data from: {_abc_path}")
        abc_df = pd.read_csv(_abc_path, sep="\t")
        sp = abc_df[abc_df["isSelfPromoter"] == True].copy()
        sp = sp.sort_values("distance").groupby("TargetGene").first().reset_index()
        for _, row in sp.iterrows():
            self_prm_lookup[row["TargetGene"]] = {
                "activity_base": float(row["activity_base"]),
                "hic_contact": float(row["hic_contact"]),
                "normalized_dhs": float(row.get("normalized_dhs", 0)),
                "distance_relative": float(row.get("distance", 0)),
                "name": row["name"],
                "chr": row["chr"],
                "start": int(row["start"]),
                "end": int(row["end"]),
            }
        print(f"Self-promoter entries: {len(self_prm_lookup)} genes")

    out_folder = output_dir or f"./seqs2expr_dataset/{cell_type}_{max_distance//1000}kb{n_enhancer}e"
    signal_files = signal_files if signal_files is not None else []
    n_sig = len(signal_files)
    os.makedirs(out_folder, exist_ok=True)
    print(f"Output: {out_folder}")

    import kipoiseq
    from Bio.Seq import Seq

    ensid_list = list(set(gene_enhancer_expr_sub_df["ENSID"]))

    # ── Pass 1: collect unique enhancers and encode sequences ──────────
    unique_enh_names = list(gene_enhancer_expr_sub_df["name"].dropna().unique())
    if include_self_promoter:
        for ensid_sp, sp_info in self_prm_lookup.items():
            sp_name = sp_info["name"]
            if sp_name not in set(unique_enh_names):
                unique_enh_names.append(sp_name)

    enh_name_to_idx = {n: i for i, n in enumerate(unique_enh_names)}
    n_unique_enh = len(unique_enh_names)
    print(f"Unique enhancers: {n_unique_enh} (incl. self-promoter: {include_self_promoter}), genes: {len(ensid_list)}")

    enh_coords = (
        gene_enhancer_expr_sub_df[["name", "chrom_x", "start_element", "end_element"]]
        .dropna(subset=["name", "start_element"])
        .drop_duplicates(subset=["name"])
        .set_index("name")
    )
    if include_self_promoter:
        sp_rows = []
        for sp_info in self_prm_lookup.values():
            sp_rows.append({
                "name": sp_info["name"],
                "chrom_x": sp_info["chr"].replace("chr", ""),
                "start_element": sp_info["start"],
                "end_element": sp_info["end"],
            })
        if sp_rows:
            sp_df = pd.DataFrame(sp_rows).drop_duplicates(subset=["name"]).set_index("name")
            enh_coords = pd.concat([enh_coords, sp_df[~sp_df.index.isin(enh_coords.index)]])

    signal_objs = []
    if n_sig > 0:
        import pyBigWig
        for sf in signal_files:
            signal_objs.append(pyBigWig.open(sf, "r"))

    h5_path = os.path.join(out_folder, "samples.h5")
    hf = create_pe_arrays_h5(
        h5_path,
        n_samples=len(ensid_list),
        n_enhancers=n_unique_enh,
        max_n_enhancer=n_enhancer,
        max_seq_len=max_seq_len,
        n_signal_tracks=n_sig,
    )

    print("Pass 1: encoding unique enhancer sequences...")
    for enh_name, enh_idx in tqdm.tqdm(enh_name_to_idx.items(), total=n_unique_enh):
        if enh_name not in enh_coords.index:
            continue
        row = enh_coords.loc[enh_name]
        chrom = "chr" + str(int(row["chrom_x"]) if not isinstance(row["chrom_x"], str) else row["chrom_x"])
        enh_start = int(row["start_element"])
        enh_end = int(row["end_element"])
        enh_center = (enh_start + enh_end) // 2
        enh_len = enh_end - enh_start

        enh_seq = np.zeros((max_seq_len, 4), dtype=np.float32)
        if add_flank or enh_len > max_seq_len:
            interval = kipoiseq.Interval(chrom, enh_center - max_seq_len // 2, enh_center + max_seq_len // 2)
            enh_seq[:] = one_hot_encode(fasta_extractor.extract(interval))
        else:
            code_start = max_seq_len // 2 - enh_len // 2
            interval = kipoiseq.Interval(chrom, enh_start, enh_end)
            enh_seq[code_start:code_start + enh_len] = one_hot_encode(fasta_extractor.extract(interval))

        enh_signal = None
        if n_sig > 0:
            enh_signal = np.zeros((max_seq_len, n_sig), dtype=np.float32)
            for s_i, sig in enumerate(signal_objs):
                if add_flank or enh_len > max_seq_len:
                    sv = sig.values(chrom, enh_center - max_seq_len // 2, enh_center + max_seq_len // 2, numpy=True)
                    enh_signal[:, s_i] = np.nan_to_num(sv)
                else:
                    sv = sig.values(chrom, enh_start, enh_end, numpy=True)
                    enh_signal[code_start:code_start + enh_len, s_i] = np.nan_to_num(sv)

        write_enhancer(hf, enh_idx, enh_name, enh_seq, signal=enh_signal)

    # ── Pass 2: write gene-level data with enhancer indices ────────────
    print("Pass 2: writing gene-level data...")
    ensid_pair_list = []
    for row_i, ensid in enumerate(tqdm.tqdm(ensid_list)):
        gene_df = gene_enhancer_expr_sub_df[
            gene_enhancer_expr_sub_df["ENSID"] == ensid
        ].sort_values(by="distance_relative")

        row_0 = gene_df.iloc[0]
        gene_tss = row_0["TSS_y"]
        chrom = "chr" + str(row_0["chrom_y"])

        target_interval = kipoiseq.Interval(
            chrom, int(gene_tss - max_seq_len / 2), int(gene_tss + max_seq_len / 2)
        )
        promoter_seq = Seq(fasta_extractor.extract(target_interval))
        promoter_code = one_hot_encode(str(promoter_seq))

        prm_signal = None
        if n_sig > 0:
            prm_signal = np.zeros((max_seq_len, n_sig), dtype=np.float32)
            for s_i, sig in enumerate(signal_objs):
                sv = sig.values(chrom, int(gene_tss - max_seq_len / 2), int(gene_tss + max_seq_len / 2), numpy=True)
                prm_signal[:, s_i] = np.nan_to_num(sv)

        gene_pe = gene_df[
            (gene_df["distance"] > _min_dist) & (gene_df["distance"] <= max_distance)
        ]

        enh_indices = np.full(n_enhancer, -1, dtype=np.int32)
        act_arr = np.zeros(n_enhancer, dtype=np.float32)
        dhs_arr = np.zeros(n_enhancer, dtype=np.float32)
        dist_arr = np.zeros(n_enhancer, dtype=np.float32)
        contact_arr = np.zeros(n_enhancer, dtype=np.float32)

        e_i = 0
        if include_self_promoter and ensid in self_prm_lookup:
            sp = self_prm_lookup[ensid]
            enh_indices[0] = enh_name_to_idx[sp["name"]]
            act_arr[0] = sp["activity_base"]
            dhs_arr[0] = sp["normalized_dhs"]
            dist_arr[0] = sp["distance_relative"]
            contact_arr[0] = sp["hic_contact"]
            ensid_pair_list.append([ensid, sp["name"]])
            e_i = 1

        for _, erow in gene_pe.iterrows():
            if pd.isna(erow["start_element"]):
                continue
            if e_i >= n_enhancer:
                break
            ename = erow["name"]
            if ename in enh_name_to_idx:
                enh_indices[e_i] = enh_name_to_idx[ename]
                act_arr[e_i] = erow["activity_base"]
                dhs_arr[e_i] = erow["normalized_dhs"]
                dist_arr[e_i] = erow["distance_relative"]
                contact_arr[e_i] = erow["hic_contact"]
                ensid_pair_list.append([ensid, ename])
                e_i += 1

        write_gene_sample(
            hf, row_i, str(ensid), promoter_code, enh_indices,
            act_arr, dhs_arr, dist_arr, contact_arr,
            promoter_signal=prm_signal,
        )

    for sig in signal_objs:
        sig.close()
    hf.close()
    gene_enhancer_df = pd.DataFrame(ensid_pair_list, columns=["ensid", "element"])
    gene_enhancer_df.to_csv(out_folder + "/gene_enhancer_pair.csv", index=False)
    gene_enhancer_expr_sub_df.to_csv(out_folder + "/gene_enhancer_pair_all.csv", index=False)


def obtain_K562_PE_withSignals(
    fname,
    max_distance=150_000,
    min_distance=0,
    add_flank=False,
    n_enhancer=60,
    max_seq_len=2000,
    gene_expression_csv: Optional[str] = None,
    fasta_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    signal_files: Optional[list[str]] = None,
    tss_column: str = "TSS",
    include_self_promoter: bool = False,
    abc_all_putative: Optional[str] = None,
):
    pe_fname = fname[0]
    enhancer_fname = fname[1]
    promoter_enhancer_df = pd.read_csv(pe_fname, sep="\t")
    promoter_enhancer_df["element_mid"] = (
        promoter_enhancer_df["end"] + promoter_enhancer_df["start"]
    ) / 2
    promoter_enhancer_df["distance_relative"] = (
        promoter_enhancer_df["element_mid"] - promoter_enhancer_df["TargetGeneTSS"]
    )
    enhancer_df = pd.read_csv(enhancer_fname, sep="\t")[
        ["name", "normalized_h3K27ac", "normalized_dhs"]
    ]
    promoter_enhancer_df = promoter_enhancer_df.merge(enhancer_df, on="name", suffixes=["_pred", "_enh"])
    # Use EnhancerList values; resolve suffix conflicts from predictions TSV also having these columns
    for col in ["normalized_h3K27ac", "normalized_dhs"]:
        if col + "_enh" in promoter_enhancer_df.columns:
            promoter_enhancer_df[col] = promoter_enhancer_df[col + "_enh"]
            promoter_enhancer_df.drop(columns=[col + "_pred", col + "_enh"], inplace=True)
    print(promoter_enhancer_df.columns)
    prefix_name = pe_fname.split("/")[-2].split("_noCutOff_")[0]

    _ge = gene_expression_csv or _DEFAULT_GENE_EXPR_CSV
    cage_xpresso_df = pd.read_csv(_ge)
    cage_xpresso_df = _apply_tss_column(cage_xpresso_df, tss_column)
    cage_xpresso_df = _orf_density_alias(cage_xpresso_df)
    cage_xpresso_df = cage_xpresso_df[
        [
            "ENSID",
            "chrom",
            "start",
            "end",
            "strand",
            "TSS",
            "Gene name",
            "K562_CAGE_128*3_sum",
            "Actual_K562",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY",
        ]
    ]
    promoter_enhancer_df = _map_symbol_to_ensid(promoter_enhancer_df, cage_xpresso_df)
    gene_enhancer_expr_df = promoter_enhancer_df.merge(
        cage_xpresso_df,
        left_on="TargetGene_ENSID",
        right_on="ENSID",
        how="right",
        suffixes=["_element", "_gene"],
    ).sort_values(by=["ENSID", "distance"])
    gene_enhancer_expr_df = gene_enhancer_expr_df[
        [
            "chrom",
            "start_element",
            "end_element",
            "hic_contact",
            "activity_base",
            "normalized_dhs",
            "name",
            "distance",
            "distance_relative",
            "ENSID",
            "TSS",
            "strand",
            "Gene name",
            "K562_CAGE_128*3_sum",
            "Actual_K562",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY",
        ]
    ]
    for distanceToTSS in [
        50_000,
        100_000,
        120_000,
        150_000,
        250_000,
        350_000,
        400_000,
        500_000,
        600_000,
    ]:
        print(distanceToTSS)
        count_df = gene_enhancer_expr_df[
            (gene_enhancer_expr_df["distance"] <= distanceToTSS)
            & (gene_enhancer_expr_df["distance"] > max_seq_len / 2)
        ].groupby("ENSID")["name"].count()
        print(
            "90%",
            np.percentile(count_df, 90),
            "95%",
            np.percentile(count_df, 95),
            np.percentile(count_df, 100),
        )

    _fa = fasta_path or _DEFAULT_FASTA_PINELLO
    fasta_extractor = FastaStringExtractor(_fa)
    _min_dist = max(min_distance, max_seq_len / 2)  # never closer than half the window
    gene_enhancer_expr_sub_df = gene_enhancer_expr_df[
        (gene_enhancer_expr_df["distance"] <= max_distance)
        & (gene_enhancer_expr_df["distance"] > _min_dist)
    ]
    gene_tss_df = gene_enhancer_expr_df.drop_duplicates(subset=["ENSID"])[
        ["ENSID", "TSS", "chrom", "strand"]
    ].reset_index()
    gene_enhancer_expr_sub_df = gene_enhancer_expr_sub_df.merge(
        gene_tss_df, on="ENSID", how="right", suffixes=["_x", "_y"]
    )

    # ── Self-promoter lookup ───────────────────────────────────────────
    self_prm_lookup = {}  # ENSID -> {activity_base, hic_contact, normalized_dhs, distance_relative, name, chr, start, end}
    if include_self_promoter:
        _abc_path = abc_all_putative or pe_fname
        print(f"Loading self-promoter data from: {_abc_path}")
        abc_df = pd.read_csv(_abc_path, sep="\t")
        sp = abc_df[abc_df["isSelfPromoter"] == True].copy()
        sp = sp.sort_values("distance").groupby("TargetGene").first().reset_index()
        for _, row in sp.iterrows():
            self_prm_lookup[row["TargetGene"]] = {
                "activity_base": float(row["activity_base"]),
                "hic_contact": float(row["hic_contact"]),
                "normalized_dhs": float(row.get("normalized_dhs", 0)),
                "distance_relative": float(row.get("distance", 0)),
                "name": row["name"],
                "chr": row["chr"],
                "start": int(row["start"]),
                "end": int(row["end"]),
            }
        print(f"Self-promoter entries: {len(self_prm_lookup)} genes")

    if output_dir is not None:
        out_folder = output_dir
    elif add_flank:
        out_folder = (
            "./seqs2expr_dataset/K562_"
            + prefix_name
            + "_"
            + str(int(max_seq_len / 1000))
            + "kb_noCutOff_hic_FlankSeq_"
            + str(int(max_distance / 1000))
            + "kb"
            + str(n_enhancer)
            + "e_AllPutative_signals"
        )
    else:
        out_folder = (
            "./seqs2expr_dataset/K562_"
            + prefix_name
            + "_"
            + str(int(max_seq_len / 1000))
            + "kb_noCutOff_hic_noFlankSeq_"
            + str(int(max_distance / 1000))
            + "kb"
            + str(n_enhancer)
            + "e_AllPutative_signals"
        )

    signal_files = signal_files if signal_files is not None else list(_DEFAULT_K562_SIGNAL_FILES)
    n_sig = len(signal_files)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print(out_folder)

    import kipoiseq
    from Bio.Seq import Seq

    ensid_list = list(set(gene_enhancer_expr_sub_df["ENSID"]))

    # ── Pass 1: collect unique enhancers and encode sequences ──────────
    unique_enh_names = list(gene_enhancer_expr_sub_df["name"].dropna().unique())

    # Add self-promoter elements as additional unique enhancers
    if include_self_promoter:
        for ensid_sp, sp_info in self_prm_lookup.items():
            sp_name = sp_info["name"]
            if sp_name not in set(unique_enh_names):
                unique_enh_names.append(sp_name)

    enh_name_to_idx = {n: i for i, n in enumerate(unique_enh_names)}
    n_unique_enh = len(unique_enh_names)
    print(f"Unique enhancers: {n_unique_enh} (incl. self-promoter: {include_self_promoter}), genes: {len(ensid_list)}")

    # Build a lookup from enhancer name → (chrom, start, end) for FASTA extraction
    enh_coords = (
        gene_enhancer_expr_sub_df[["name", "chrom_x", "start_element", "end_element"]]
        .dropna(subset=["name", "start_element"])
        .drop_duplicates(subset=["name"])
        .set_index("name")
    )
    # Add self-promoter coordinates to enh_coords
    if include_self_promoter:
        sp_rows = []
        for sp_info in self_prm_lookup.values():
            sp_rows.append({
                "name": sp_info["name"],
                "chrom_x": sp_info["chr"].replace("chr", ""),
                "start_element": sp_info["start"],
                "end_element": sp_info["end"],
            })
        if sp_rows:
            sp_df = pd.DataFrame(sp_rows).drop_duplicates(subset=["name"]).set_index("name")
            enh_coords = pd.concat([enh_coords, sp_df[~sp_df.index.isin(enh_coords.index)]])

    signal_objs = []
    if n_sig > 0:
        import pyBigWig
        for sf in signal_files:
            signal_objs.append(pyBigWig.open(sf, "r"))

    h5_path = os.path.join(out_folder, "samples.h5")
    hf = create_pe_arrays_h5(
        h5_path,
        n_samples=len(ensid_list),
        n_enhancers=n_unique_enh,
        max_n_enhancer=n_enhancer,
        max_seq_len=max_seq_len,
        n_signal_tracks=n_sig,
    )

    print("Pass 1: encoding unique enhancer sequences...")
    for enh_name, enh_idx in tqdm.tqdm(enh_name_to_idx.items(), total=n_unique_enh):
        if enh_name not in enh_coords.index:
            continue
        row = enh_coords.loc[enh_name]
        chrom = "chr" + str(int(row["chrom_x"]) if not isinstance(row["chrom_x"], str) else row["chrom_x"])
        enh_start = int(row["start_element"])
        enh_end = int(row["end_element"])
        enh_center = (enh_start + enh_end) // 2
        enh_len = enh_end - enh_start

        # Encode sequence (same logic as extract_promoter_enhancer_loci_signals)
        enh_seq = np.zeros((max_seq_len, 4), dtype=np.float32)
        if add_flank or enh_len > max_seq_len:
            interval = kipoiseq.Interval(chrom, enh_center - max_seq_len // 2, enh_center + max_seq_len // 2)
            enh_seq[:] = one_hot_encode(fasta_extractor.extract(interval))
        else:
            code_start = max_seq_len // 2 - enh_len // 2
            interval = kipoiseq.Interval(chrom, enh_start, enh_end)
            enh_seq[code_start:code_start + enh_len] = one_hot_encode(fasta_extractor.extract(interval))

        # Encode signal
        enh_signal = None
        if n_sig > 0:
            enh_signal = np.zeros((max_seq_len, n_sig), dtype=np.float32)
            for s_i, sig in enumerate(signal_objs):
                if add_flank or enh_len > max_seq_len:
                    sv = sig.values(chrom, enh_center - max_seq_len // 2, enh_center + max_seq_len // 2, numpy=True)
                    enh_signal[:, s_i] = np.nan_to_num(sv)
                else:
                    sv = sig.values(chrom, enh_start, enh_end, numpy=True)
                    enh_signal[code_start:code_start + enh_len, s_i] = np.nan_to_num(sv)

        write_enhancer(hf, enh_idx, enh_name, enh_seq, signal=enh_signal)

    # ── Pass 2: write gene-level data with enhancer indices ────────────
    print("Pass 2: writing gene-level data...")
    ensid_pair_list = []
    for row_i, ensid in enumerate(tqdm.tqdm(ensid_list)):
        gene_df = gene_enhancer_expr_sub_df[
            gene_enhancer_expr_sub_df["ENSID"] == ensid
        ].sort_values(by="distance_relative")

        row_0 = gene_df.iloc[0]
        gene_tss = row_0["TSS_y"]
        chrom = "chr" + str(row_0["chrom_y"])

        # Promoter sequence
        target_interval = kipoiseq.Interval(
            chrom, int(gene_tss - max_seq_len / 2), int(gene_tss + max_seq_len / 2)
        )
        promoter_seq = Seq(fasta_extractor.extract(target_interval))
        promoter_code = one_hot_encode(str(promoter_seq))

        # Promoter signal
        prm_signal = None
        if n_sig > 0:
            prm_signal = np.zeros((max_seq_len, n_sig), dtype=np.float32)
            for s_i, sig in enumerate(signal_objs):
                sv = sig.values(chrom, int(gene_tss - max_seq_len / 2), int(gene_tss + max_seq_len / 2), numpy=True)
                prm_signal[:, s_i] = np.nan_to_num(sv)

        # Filter enhancers by distance
        gene_pe = gene_df[
            (gene_df["distance"] > _min_dist) & (gene_df["distance"] <= max_distance)
        ]

        # Build per-gene enhancer index + features
        enh_indices = np.full(n_enhancer, -1, dtype=np.int32)
        act_arr = np.zeros(n_enhancer, dtype=np.float32)
        dhs_arr = np.zeros(n_enhancer, dtype=np.float32)
        dist_arr = np.zeros(n_enhancer, dtype=np.float32)
        contact_arr = np.zeros(n_enhancer, dtype=np.float32)

        # Insert self-promoter at slot 0 if available
        e_i = 0
        if include_self_promoter and ensid in self_prm_lookup:
            sp = self_prm_lookup[ensid]
            enh_indices[0] = enh_name_to_idx[sp["name"]]
            act_arr[0] = sp["activity_base"]
            dhs_arr[0] = sp["normalized_dhs"]
            dist_arr[0] = sp["distance_relative"]
            contact_arr[0] = sp["hic_contact"]
            ensid_pair_list.append([ensid, sp["name"]])
            e_i = 1

        for _, erow in gene_pe.iterrows():
            if pd.isna(erow["start_element"]):
                continue
            if e_i >= n_enhancer:
                break
            ename = erow["name"]
            if ename in enh_name_to_idx:
                enh_indices[e_i] = enh_name_to_idx[ename]
                act_arr[e_i] = erow["activity_base"]
                dhs_arr[e_i] = erow["normalized_dhs"]
                dist_arr[e_i] = erow["distance_relative"]
                contact_arr[e_i] = erow["hic_contact"]
                ensid_pair_list.append([ensid, ename])
                e_i += 1

        write_gene_sample(
            hf, row_i, str(ensid), promoter_code, enh_indices,
            act_arr, dhs_arr, dist_arr, contact_arr,
            promoter_signal=prm_signal,
        )

    for sig in signal_objs:
        sig.close()
    hf.close()
    gene_enhancer_df = pd.DataFrame(ensid_pair_list, columns=["ensid", "element"])
    gene_enhancer_df.to_csv(out_folder + "/gene_enhancer_pair.csv", index=False)
    gene_enhancer_expr_sub_df.to_csv(out_folder + "/gene_enhancer_pair_all.csv", index=False)


def obtain_GM12878_PE_withSignals(
    max_distance=150_000, add_flank=True, max_seq_len=1000
):
    promoter_enhancer_df = pd.read_csv(
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/GM12878_data/DNase_ENCFF257HEE_noCutOff_hic_1MB_predictions/EnhancerPredictionsAllPutative.txt.gz",
        sep="\t",
    )
    cage_xpresso_df = pd.read_csv(
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/GM12878_K562_18377_gene_expression.csv"
    )
    cage_xpresso_df = cage_xpresso_df[
        [
            "ENSID",
            "chrom",
            "start",
            "end",
            "strand",
            "TSS",
            "Gene name",
            "K562_CAGE_128*3_sum",
            "Actual_K562",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY",
        ]
    ]
    promoter_enhancer_df = _map_symbol_to_ensid(promoter_enhancer_df, cage_xpresso_df)
    gene_enhancer_expr_df = promoter_enhancer_df.merge(
        cage_xpresso_df,
        left_on="TargetGene_ENSID",
        right_on="ENSID",
        how="right",
        suffixes=["_element", "_gene"],
    ).sort_values(by=["ENSID", "distance"])
    gene_enhancer_expr_df = gene_enhancer_expr_df[
        [
            "chrom",
            "start_element",
            "end_element",
            "hic_contact",
            "activity_base",
            "name",
            "distance",
            "ENSID",
            "TSS",
            "strand",
            "Gene name",
            "K562_CAGE_128*3_sum",
            "Actual_K562",
            "UTR5LEN_log10zscore",
            "CDSLEN_log10zscore",
            "INTRONLEN_log10zscore",
            "UTR3LEN_log10zscore",
            "UTR5GC",
            "CDSGC",
            "UTR3GC",
            "ORFEXONDENSITY",
        ]
    ]
    for distanceToTSS in [
        50_000,
        100_000,
        150_000,
        250_000,
        350_000,
        400_000,
        500_000,
        600_000,
    ]:
        print(distanceToTSS)
        count_df = gene_enhancer_expr_df[
            gene_enhancer_expr_df["distance"] <= distanceToTSS
        ].groupby("ENSID")["name"].count()
        print(
            "90%",
            np.percentile(count_df, 90),
            "95%",
            np.percentile(count_df, 95),
            np.percentile(count_df, 100),
        )

    fasta_extractor = FastaStringExtractor(
        "/data/pinello/PROJECTS/2020_06_Pred_Gene_expression/EPInformerV2_20230209//hg38.fa"
    )
    gene_enhancer_expr_sub_df = gene_enhancer_expr_df[
        (gene_enhancer_expr_df["distance"] <= max_distance)
        & (gene_enhancer_expr_df["distance"] > 500)
    ]
    gene_tss_df = gene_enhancer_expr_df.drop_duplicates(subset=["ENSID"])[
        ["ENSID", "TSS", "chrom", "strand"]
    ].reset_index()
    gene_enhancer_expr_sub_df = gene_enhancer_expr_sub_df.merge(
        gene_tss_df, on="ENSID", how="right", suffixes=["_x", "_y"]
    )

    if add_flank:
        out_folder = "./seqs2expr_dataset/K562_DNase_ENCFF257HEE_1kbFlankSeq_noCutOff_hic_150kb60e_AllPutative_multiSignals"
    else:
        out_folder = "./seqs2expr_dataset/K562_DNase_ENCFF257HEE_1kbNoFlankSeq_noCutOff_hic_150kb60e_AllPutative_multiSignals"
    signal_files = [
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/DNase/ENCFF972GVB.bigWig",
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/H3K27ac/ENCFF465GBD.bigWig",
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/H3K4me1/ENCFF287LBI.bigWig",
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/H3K4me3/ENCFF144MRB.bigWig",
        "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/CTCF/ENCFF168IFW.bigWig",
    ]
    n_sig = len(signal_files)
    n_enh = 60
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print(out_folder)
    ensid_pair_list = []
    ensid_list = list(set(gene_enhancer_expr_sub_df["ENSID"]))
    h5_path = os.path.join(out_folder, "samples.h5")
    hf = create_pe_arrays_h5(
        h5_path,
        n_samples=len(ensid_list),
        max_n_enhancer=n_enh,
        max_seq_len=max_seq_len,
        n_signal_tracks=n_sig,
    )
    for row_i, ensid in enumerate(tqdm.tqdm(ensid_list)):
        gene_df = gene_enhancer_expr_sub_df[gene_enhancer_expr_sub_df["ENSID"] == ensid]
        (
            seq_code,
            act_list,
            dhs_list,
            dist_list,
            contact_list,
            _ens,
            ensid_element_pair,
            seq_signal,
        ) = extract_promoter_enhancer_loci_signals(
            gene_df,
            fasta_extractor,
            signal_file=signal_files,
            max_seq_len=max_seq_len,
            max_n_enhancer=n_enh,
            max_distanceToTSS=max_distance,
            add_flanking=add_flank,
        )
        ensid_pair_list = ensid_pair_list + ensid_element_pair
        write_pe_sample(
            hf,
            row_i,
            str(ensid),
            seq_code,
            act_list,
            dhs_list,
            dist_list,
            contact_list,
            seq_signal,
            n_signal_tracks=n_sig,
        )
    hf.close()
    gene_enhancer_df = pd.DataFrame(ensid_pair_list, columns=["ensid", "element"])
    gene_enhancer_df.to_csv(out_folder + "/gene_enhancer_pair.csv", index=False)
    gene_enhancer_expr_sub_df.to_csv(out_folder + "/gene_enhancer_pair_all.csv", index=False)


# Example driver block (uncomment paths and run as a script; not executed on import)
if __name__ == "__main__":
    pass
    # fname = [
    #     "/data/pinello/PROJECTS/2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/DNase_ENCFF257HEE_hic_noCutOff_1MB_predictions/EnhancerPredictionsAllPutative.txt.gz",
    #     "/data/pinello/PROJECTS//2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/DNase_ENCFF257HEE_Neighborhoods/EnhancerList.txt",
    # ]
    # signal_files = [
    #     "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/DNase/ENCFF972GVB.bigWig",
    #     "../2020_06_Pred_Gene_expression/EPInformerV2_20230209/K562_data/H3K27ac/ENCFF465GBD.bigWig",
    # ]
    # obtain_PE(
    #     fname,
    #     signal_files,
    #     max_seq_len=2000,
    #     n_enhancer=60,
    #     add_flank=False,
    #     max_distance=100_000,
    #     cell_type="K562",
    # )
