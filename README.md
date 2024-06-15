<p align="center">
  <img height="560" src="images/EPInformer.png">
</p>

Welcome to the EPInformer framework repository! EPInformer is a scalable deep learning framework for gene expression prediction by integrating promoter-enhancer sequences with epigenomic signals. EPInformer is designed for three key applications: 1) predict gene expression levels using promoter-enhancer sequences, epigenomic signals, and chromatin contacts; 2) identify cell-type-specific enhancer-gene interactions, validated by CRISPR perturbation experiments; 3) predict enhancer activity and recapitulate transcription factor binding motifs from sequences.

This repository can be used to run the EPInformer model to predit gene expression and prioritize enhancer-gene interactions for input DNA sequences and epigenomic signals (e.g. DNase, H3K27ac and Hi-C).

We also provide information and instructions for how to train different versions of EPInformer given diffenet inputs including DNA sequence, epigemoic signals and chromatine contacts.

### Requirements

EPInformer requires Python 3.6+ and Python packages PyTorch (>=2.1). You can follow PyTorch installation steps [here](https://pytorch.org/get-started/locally/).

### Setup

EPInformer requires ABC enhancer-gene data for training and predicting gene expression. You can obtain the ABC data from [ENCODE](https://www.encodeproject.org/search/?type=Annotation&annotation_type=element+gene+regulatory+interaction+predictions&software_used.software.name=abc-enhancer-gene-prediction-encode_v1) or by running the ABC pipeline available on their [GitHub](https://github.com/broadinstitute/ABC-Enhancer-Gene-Prediction) acquire cell-type-specific gene-enhancer links. We provide a script [here](https://github.com/JasonLinjc/EPInformer/tree/main/data) for downloading ABC enhancer-gene links from ENCODE for *K562* and *GM12878* cells.

### Gene expression prediction
To predict the gene expression measured by CAGE-seq or RNA-seq in *K562* and *GM12878* cells with EPInformer, please first run the folloing command to setup the environment:
```
# Clone this repository
git clone https://github.com/JasonLinjc/EPInformer.git
cd EPInformer

# download the pre-trained EPInformer models from zenodo (coming soon)

# create 'EPInformer_env' conda environment by running the following:
conda create --name EPInformer_env python=3.8 pandas scipy scikit-learn jupyter seaborn
source activate EPInformer_env

# GPU version putorch
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU version pytorch
conda install pytorch cpuonly -c pytorch

# Other pacakges
pip install pyranges pyfaidx kipoiseq openpyxl
```
An end-to-end example to predict gene expression from promoter-enhancer links is in [1_predict_gene_expression.ipynb](https://github.com/JasonLinjc/EPInformer/blob/main/1_predict_gene_expression.ipynb). You can run this notebook yourself to experiment with different EPInformers.

### Enhancer-gene links prediction
To prioritize the enhancer-gene links tested by [CRISPRi-FlowFISH](https://www.nature.com/articles/s41588-019-0538-0) in *K562*, we obtain the original data from their [supplementray table](https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-019-0538-0/MediaObjects/41588_2019_538_MOESM3_ESM.xlsx). We provide a jupyter notebook [(2_prioritize_enhancer_gene_links.ipynb)](https://github.com/JasonLinjc/EPInformer/blob/main/2_prioritize_enhancer_gene_links.ipynb) for pre-processing CRISPRi-FlowFISH data and scoring enhancer-gene links using EPInformer-derived attention scores and the Attention-ABC score. The notebook includes an end-to-end example of performing in-silico perturbations on each candidate element within 100kb of KLF1 and predicting their effects.

<p align="center">
  <img height="800" src="images/KLF1_insilico_perturbation.png">
</p>

### Enhancer activity prediction and TF motif discovery
