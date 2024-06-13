from kipoiseq import Interval
import pyfaidx
import kipoiseq
import numpy as np
import pandas as pd
import pyranges as pr
# from Bio.Seq import Seq
from tqdm import tqdm

def df_to_pyranges(df, start_col='start', end_col='end', chr_col='chr', start_slop=0, end_slop=0):
    df['Chromosome'] = df[chr_col]
    df['Start'] = df[start_col] - start_slop
    df['End'] = df[end_col] + end_slop
    return(pr.PyRanges(df))

class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()
    
def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

def encode_promoter_enhancer_links(gene_enhancer_df, fasta_path = '../hg38.fa', max_n_enhancer = 60, max_distanceToTSS = 100_000, max_seq_len=2000, add_flanking=False):
    fasta_extractor = FastaStringExtractor(fasta_path)
    gene_pe = gene_enhancer_df.sort_values(by='distance')
    row_0 = gene_pe.iloc[0]
    gene_name = row_0['TargetGene']
    gene_tss = row_0['TargetGeneTSS']
    chrom = row_0['#chr']
    target_interval = kipoiseq.Interval(chrom, int(gene_tss-max_seq_len/2), int(gene_tss+max_seq_len/2))

    promoter_seq = fasta_extractor.extract(target_interval)
    promoter_code = one_hot_encode(promoter_seq)
    enhancers_code = np.zeros((max_n_enhancer, max_seq_len, 4))
    enhancer_activity = np.zeros(max_n_enhancer)
    enhancer_distance = np.zeros(max_n_enhancer)
    enhancer_contact = np.zeros(max_n_enhancer)
    # set distance threshold
    gene_pe = gene_pe[(gene_pe['distance'] > max_seq_len/2)&(gene_pe['distance'] <= max_distanceToTSS)]
    e_i = 0
    gene_element_pair = []
    for idx, row in gene_pe.iterrows():
        if pd.isna(row['start']):
            continue
        if e_i >= max_n_enhancer:
            break
        enhancer_start = int(row['start'])
        enhancer_end = int(row['end'])
        enhancer_center = int((row['start'] + row['end'])/2)
        enhancer_len = enhancer_end - enhancer_start
        # put sequence at the center
        if add_flanking:
            enhancer_target_interval = kipoiseq.Interval(chrom, enhancer_center-int(max_seq_len/2), enhancer_center+int(max_seq_len/2))
            enhancers_code[e_i][:] = one_hot_encode(fasta_extractor.extract(enhancer_target_interval))
        else:
            # enhancers_signal = np.zeros((max_n_enhancer, max_seq_len))
            if enhancer_len > max_seq_len:
                enhancer_target_interval = kipoiseq.Interval(chrom, enhancer_center-int(max_seq_len/2), enhancer_center+int(max_seq_len/2))
                enhancers_code[e_i][:] = one_hot_encode(fasta_extractor.extract(enhancer_target_interval))
            else:
                code_start = int(max_seq_len/2)-int(enhancer_len/2)
                enhancer_target_interval = kipoiseq.Interval(chrom, enhancer_start, enhancer_end)
                enhancers_code[e_i][code_start:code_start+enhancer_len] = one_hot_encode(fasta_extractor.extract(enhancer_target_interval))
        # put sequence from the start
        enhancer_activity[e_i] = row['activity_base']
        enhancer_distance[e_i] = row['distance']
        enhancer_contact[e_i] = row['hic_contact']*100
        gene_element_pair.append([gene_name, row['name']])
        e_i += 1
    # print(promoter_signals.shape, enhancers_signal.shape)
    pe_code = np.concatenate([promoter_code[np.newaxis,:], enhancers_code], axis=0)
    gene_element_pair = pd.DataFrame(gene_element_pair, columns=['gene', 'element'])
    return pe_code, enhancer_activity, enhancer_distance, enhancer_contact, gene_name, gene_element_pair


def prepare_input(gene_enhancer_table, gene_list, num_features = 3):
    # enhancer_gene_k562_100kb[enhancer_gene_k562_100kb['#chr'] == 'chrX']['TargetGene'].unique()
    mRNA_feauture = pd.read_csv('./data/mRNA_halflife_features.csv', index_col='Gene name')
    promoter_signals = pd.read_csv('./data/GeneList_K562.txt', index_col='symbol', sep='\t')
    promoter_signals['PromoterActivity'] = np.sqrt(promoter_signals['H3K27ac.RPM.TSS1Kb']*promoter_signals['DHS.RPM.TSS1Kb'])
    mRNA_feats = ['UTR5LEN_log10zscore',
       'CDSLEN_log10zscore', 'INTRONLEN_log10zscore', 'UTR3LEN_log10zscore',
       'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY']
    PE_code_list = []
    PE_feat_list = []
    mRNA_promoter_list = []
    PE_links_list = []
    for gene in tqdm(gene_list):
        gene_df = gene_enhancer_table[gene_enhancer_table['TargetGene'] == gene]
        PE_code, activity_list, distance_list, contact_list, gene_name, PE_links = encode_promoter_enhancer_links(gene_df, max_seq_len=2000, max_n_enhancer=60, max_distanceToTSS=100_000, add_flanking=False)
        contact_list = np.concatenate([[0], contact_list*100])
        distance_list = np.concatenate([[0], distance_list/1000])
        activity_list = np.concatenate([[0], activity_list])
        activity_list = np.log10(0.1+activity_list)
        mRNA_promoter_feat = np.array(list(mRNA_feauture.loc[gene, mRNA_feats].values) + [promoter_signals.loc[gene, 'PromoterActivity']])
        if num_features == 1:
            PE_feat = distance_list[:,np.newaxis]
            mRNA_promoter_feat = np.array(list(mRNA_feauture.loc[gene, mRNA_feats].values) + [0])
        elif num_features == 2:
            PE_feat = np.concatenate([distance_list[:,np.newaxis], activity_list[:,np.newaxis], ],axis=-1)
        else:
            PE_feat = np.concatenate([distance_list[:,np.newaxis], contact_list[:,np.newaxis], activity_list[:,np.newaxis], ],axis=-1)
        # print(gene_name, PE_code.shape, PE_feat.shape, mRNA_promoter_feat.shape)
        PE_code_list.append(PE_code)
        PE_feat_list.append(PE_feat)
        mRNA_promoter_list.append(mRNA_promoter_feat)
        PE_links_list.append(PE_links)
    PE_links_df = pd.concat(PE_links_list)
    PE_code_list = np.array(PE_code_list)
    PE_feat_list = np.array(PE_feat_list)
    mRNA_promoter_list = np.array(mRNA_promoter_list)
    return PE_code_list, PE_feat_list, mRNA_promoter_list, PE_links_df