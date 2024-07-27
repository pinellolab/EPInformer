from kipoiseq import Interval
import pyfaidx
import kipoiseq
import numpy as np
import pandas as pd
import pyranges as pr
# from Bio.Seq import Seq
from tqdm import tqdm
import os
import torch

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

def encode_promoter_enhancer_links(gene_enhancer_df, fasta_path = './data/hg38.fa', max_n_enhancer = 60, max_distanceToTSS = 100_000, max_seq_len=2000, add_flanking=False):
    fasta_extractor = FastaStringExtractor(fasta_path)
    gene_pe = gene_enhancer_df.sort_values(by='distance')
    row_0 = gene_pe.iloc[0]
    gene_name = row_0['TargetGene']
    gene_tss = row_0['TargetGeneTSS']
    chrom = row_0['chr']
    if row_0['TargetGeneTSS'] != row_0['TargetGeneTSS']:
        gene_tss = row_0['tss']
        gene_name = row_0['name_gene']
        chrom = row_0['chr']
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
        if row['TargetGene'] != row['TargetGene']:
            break
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
        enhancer_contact[e_i] = row['hic_contact']
        gene_element_pair.append([gene_name, row['name']])
        e_i += 1
    # print(promoter_signals.shape, enhancers_signal.shape)
    pe_code = np.concatenate([promoter_code[np.newaxis,:], enhancers_code], axis=0)
    gene_element_pair = pd.DataFrame(gene_element_pair, columns=['gene', 'element'])
    return pe_code, enhancer_activity, enhancer_distance, enhancer_contact, gene_name, gene_element_pair


def prepare_input(gene_enhancer_table, gene_list, cell, num_features = 3):
    # enhancer_gene_k562_100kb[enhancer_gene_k562_100kb['#chr'] == 'chrX']['TargetGene'].unique()
    mRNA_feauture = pd.read_csv('./data/mRNA_halflife_features.csv', index_col='gene_id')
    if cell == 'K562':
        promoter_signals = pd.read_csv('./data/K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt', sep='\t', index_col='symbol')
        promoter_signals['PromoterActivity'] = np.sqrt(promoter_signals['H3K27ac.RPM.TSS1Kb']*promoter_signals['DHS.RPM.TSS1Kb'])
    elif cell == 'GM12878':
        promoter_signals = pd.read_csv('./data/GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated/DNase_ENCFF020WZB_Neighborhoods/GeneList.txt', sep='\t', index_col='symbol')
        promoter_signals['PromoterActivity'] = np.sqrt(promoter_signals['H3K27ac.RPM.TSS1Kb']*promoter_signals['DHS.RPM.TSS1Kb'])
    else:
        print(cell, 'not found!')
        return 0
    mRNA_feats = ['UTR5LEN_log10zscore',
       'CDSLEN_log10zscore', 'INTRONLEN_log10zscore', 'UTR3LEN_log10zscore',
       'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY']
    PE_code_list = []
    PE_feat_list = []
    mRNA_promoter_list = []
    PE_links_list = []
    for gene in tqdm(gene_list):
        gene_df = gene_enhancer_table[gene_enhancer_table['ENSID'] == gene]
        PE_code, activity_list, distance_list, contact_list, gene_name, PE_links = encode_promoter_enhancer_links(gene_df, max_seq_len=2000, max_n_enhancer=60, max_distanceToTSS=100_000, add_flanking=False)
        contact_list = np.concatenate([[0], contact_list])
        distance_list = np.concatenate([[0], distance_list/1000])
        activity_list = np.concatenate([[0], activity_list])
        # activity_list = np.log10(0.1+activity_list)
        contact_list = np.log10(1+contact_list)
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

def encoder_promoter_enhancer_CRISPRi(pe_df, hg19_fasta_path = './data/hg19.fa', verbose=True, HiC_norm=False):
    pe_df = pe_df.sort_values(by='Distance')
    if 'level_0' in pe_df.columns:
        pe_df.drop(columns=['level_0'], inplace=True)
    if not os.path.exists(hg19_fasta_path):
        print('Downloading hg19 reference genome...')
        import urllib.request
        urllib.request.urlretrieve("https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz", hg19_fasta_path+".gz")
        os.system('gunzip ' +  hg19_fasta_path+".gz")
        # hg19_fasta_path = './data/hg19.fa'
    hg19_fasta_extractor = FastaStringExtractor(hg19_fasta_path)
    RNA_feats = pd.read_csv('./data/GM12878_K562_18377_gene_expr_fromXpresso.csv', index_col='Gene stable ID')[['UTR5LEN_log10zscore','CDSLEN_log10zscore','INTRONLEN_log10zscore','UTR3LEN_log10zscore','UTR5GC','CDSGC','UTR3GC', 'ORFEXONDENSITY_log10zscore']]
    promoter_df = pd.read_csv('./data/CRISPRi-FlowFISH_Fulco2019/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt', sep='\t', index_col='symbol')
    promoter_df['PromoterActivity'] = np.sqrt(promoter_df['H3K27ac.RPM.TSS1Kb']*promoter_df['DHS.RPM.TSS1Kb'])
    max_n_enhancer = len(pe_df)
    max_seq_len = 2000
    enhancer_distance = np.zeros(max_n_enhancer)
    enhancer_activity = np.zeros(max_n_enhancer)
    enhancer_hic = np.zeros(max_n_enhancer)
    enhancers_code = np.zeros((max_n_enhancer, max_seq_len, 4))
    e_i = 0
    eid = pe_df.iloc[0]['Gene stable ID']
    singleGene_promoter_activity = promoter_df.loc[eid, 'PromoterActivity']

    gene_tss = pe_df.iloc[0]['Gene TSS']
    gene_name = pe_df.iloc[0]['Gene name']

    chrom = pe_df.iloc[0]['chr']
    target_interval = kipoiseq.Interval(chrom, int(gene_tss-max_seq_len/2), int(gene_tss+max_seq_len/2))
    if verbose:
        print(eid, gene_name, chrom)
    promoter_seq = hg19_fasta_extractor.extract(target_interval)
    promoter_code = one_hot_encode(str(promoter_seq))
    for idx, row in pe_df.iterrows():
        chrom = row['chr']
        enhancer_start = int(row['start'])
        enhancer_end = int(row['end'])
        # enhancer_center = int((enhancer_start + enhancer_end)/2)
        enhancer_center = int(row['mid'])
        enhancer_len = enhancer_end - enhancer_start
        if enhancer_len > 2000:
            enhancer_start = enhancer_center-1000
            enhancer_end = enhancer_center+1000
            enhancer_len = 2000
        # print(enhancer_len)
        code_start = int(max_seq_len/2)-int(enhancer_len/2)
        enhancer_target_interval = kipoiseq.Interval(chrom, enhancer_start, enhancer_end)
        enhancers_code[e_i][code_start:code_start+enhancer_len] = one_hot_encode(hg19_fasta_extractor.extract(enhancer_target_interval))
        enhancer_activity[e_i] = row['Activity']
        enhancer_distance[e_i] = abs(row['Distance'])
        enhancer_hic[e_i] = row['Normalized HiC Contacts']
        e_i += 1

    pe_code = np.concatenate([promoter_code[np.newaxis, ], enhancers_code], axis=0)   
    pe_distance = np.concatenate([[0], enhancer_distance/1000]).flatten()
    # print(rna_ts)
    pe_activity = np.concatenate([[0], enhancer_activity]).flatten()
    pe_hic = np.concatenate([[0], enhancer_hic]).flatten()
    if HiC_norm:
        pe_hic = np.log10(0.1+pe_hic*10)
    pe_activity = np.log10(0.1+pe_activity)
    rna_ts = np.array(list(RNA_feats.loc[eid].values) + [singleGene_promoter_activity])[np.newaxis,:]
    pe_feat = np.concatenate([pe_distance[:,np.newaxis], pe_activity[:,np.newaxis], pe_hic[:,np.newaxis]],axis=-1)
    return pe_code, pe_feat, rna_ts

def compute_enhancer_gene_attention(model, pe_df, use_hic=False, device='cpu'):
    if 'level_0' in pe_df.columns:
        pe_df = pe_df.drop(columns='level_0')
    pe_df = pe_df.sort_values(by='Distance').reset_index()
    model.eval()
    with torch.no_grad():
      seq_input, feat_input, rna_input = encoder_promoter_enhancer_CRISPRi(pe_df, hg19_fasta_path='../hg19.fa', verbose=False, HiC_norm=True)
      seq_input = torch.from_numpy(seq_input).unsqueeze(0).float().to(device)
      feat_input = torch.from_numpy(feat_input).unsqueeze(0).float().to(device)
      rna_input = torch.from_numpy(rna_input).float().to(device)
      if not use_hic:
          feat_input = feat_input[:, :, :2] # exclude HiC
      else:
          feat_input = torch.cat([feat_input[:,:,:1], feat_input[:,:,2:], feat_input[:,:,1:2]], dim=-1)
      # print(feat_input.shape)
      all_expr, attn_list = model(seq_input, rna_input, feat_input)
      attn_list = attn_list.permute((1, 0, 2, 3))+1e-5
      attn_meanLayer = attn_list.mean(1)[:,0].cpu().detach().numpy()[0]
      all_expr = all_expr.cpu().detach().numpy()[0][0]
    attention_mean = attn_meanLayer[1:]
    return attention_mean, all_expr

def predict_enhancer_activity(enhancer_model, chrom, position, window_size=1024, stride=128, device='cuda'):
    hg19_fasta_path = '../hg19.fa'
    if not os.path.exists(hg19_fasta_path):
        print('Downloading hg19 reference genome...')
        import urllib.request
        urllib.request.urlretrieve("https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz", "./data/hg19.fa.gz")
        os.system('gunzip ./data/hg19.fa.gz')
        hg19_fasta_path = './data/hg19.fa'
    hg19_fasta_extractor = FastaStringExtractor(hg19_fasta_path)
    center = position
    print('The extened enhancer region: {}:{}-{}'.format(chrom, center-window_size, center+window_size))
    pred_list = []
    x_list = []
    # window_size = 3000
    # stride = 200
    info_list = []
    for i in range(center-window_size, center+window_size, stride):
        x_list.append(i+stride)
        target_interval = kipoiseq.Interval(chrom, i, i+256)
        seq = hg19_fasta_extractor.extract(target_interval)
        seq_code = one_hot_encode(seq)
        seq_code = seq_code[np.newaxis,np.newaxis,:]
        seq_code_tensor = torch.Tensor(seq_code).to(device)
        enhancer_model.eval()
        with torch.no_grad():
            pred = enhancer_model(seq_code_tensor).cpu().detach().numpy()[0]
            pred_ori = 2**pred-0.1
            pred_list.append(pred_ori)
        info_list.append([chrom, i, i+256, seq, center, pred_ori])

    info_df = pd.DataFrame(info_list, columns=['chrom', 'start', 'end', 'seq', 'enhancer_mid', 'pred'])
    return info_df

def perturb_enhancer(model, pe_df, use_hic=False, device='cpu'):
    pe_df = pe_df.sort_values(by='Distance').reset_index()
    model.eval()
    perturb_pred_list = []
    with torch.no_grad():
      seq_input, feat_input, rna_input = encoder_promoter_enhancer_CRISPRi(pe_df, hg19_fasta_path='../hg19.fa')
      seq_input = torch.from_numpy(seq_input)
      feat_input = torch.from_numpy(feat_input)
      rna_input = torch.from_numpy(rna_input)
      seq_input = seq_input.unsqueeze(0).float().to(device)
      rna_input = rna_input.float().to(device)
      feat_input = feat_input.unsqueeze(0).float().to(device)
      distToTSS = feat_input[:, :, 0].detach().cpu().numpy().flatten()*1000
      signals = feat_input[:, :, 1].detach().cpu().numpy().flatten()
      hic = feat_input[:, :, 2].detach().cpu().numpy().flatten()
      if not use_hic:
          feat_input = feat_input[:, :, :2] # exclude HiC
      else:
          feat_input = torch.cat([feat_input[:,:,:1], feat_input[:,:,2:], feat_input[:,:,1:2]], dim=-1)
      # print(feat_input.shape)
      all_expr, attn_list = model(seq_input, rna_input, feat_input)
      all_expr = all_expr.cpu().detach().numpy()[0][0]
      all_expr = 10**all_expr-1
      # print('predictive expression', all_expr)
      attn_list = attn_list.permute((1, 0, 2, 3))+1e-5
      attn_meanLayer = attn_list.mean(1)[:,0].cpu().detach().numpy()[0]
      print('Calucuting the change of predicted expression by in-silico perturbation...')
      for mask_ei in tqdm(range(1, len(pe_df)+1)):
          # print(mask_ei)
          seq_perturb = seq_input.clone()
          seq_perturb[:,mask_ei,:, :] = torch.zeros((1, 2000, 4))
          pred_expr, _ = model(seq_perturb, rna_input, feat_input)
          pred_expr = pred_expr.cpu().detach().numpy().flatten()[0]
          pred_expr = 10**pred_expr-1
          perturb_pred_list.append(pred_expr)

    perturb_pred_list = np.array(perturb_pred_list)
    pe_df['change of predicted expression'] = (((perturb_pred_list - all_expr))/all_expr)
    
    pe_df['in-silico perturb expr'] = perturb_pred_list
    pe_df['distanceToTSS'] = (distToTSS[1:]).astype(int)
    pe_df['enhancer activity'] = 10**signals[1:]-1
    pe_df['HiC contact'] = hic[1:]
    pe_df['Attention score'] = attn_meanLayer[1:]/sum(attn_meanLayer[1:])
    pe_df['pred_expr'] = all_expr
    return pe_df