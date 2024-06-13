<p align="center">
  <img height="560" src="images/EPInformer.png">
</p>

Welcome to the EPInformer framework repository! EPInformer is a scalable deep learning framework for gene expression prediction by integrating promoter-enhancer sequences with epigenomic signals. EPInformer is designed for three key applications: 1) predict gene expression levels using promoter-enhancer sequences, epigenomic signals, and chromatin contacts; 2) identify cell-type-specific enhancer-gene interactions, validated by CRISPR perturbation experiments; 3) predict enhancer activity and recapitulate transcription factor binding motifs from sequences.

This repository can be used to run the EPInformer model and get gene expression predictions and prioritize enhancer-gene interactions for input sequences and epigenomic signals.

We also provide information and instructions for how to train different versions of EPInformer given diffenet inputs including DNA sequence, epigemoic signals and chomatine contacts.

### Requirements

EPInformer requires Python 3.6+ and Python packages PyTorch (>=2.1). You can follow PyTorch installation steps [here](https://pytorch.org/get-started/locally/).

### Setup

EPInformer requires ABC enhancer-gene data for training and predicting gene expression. You can obtain the ABC data from [ENCODE](https://www.encodeproject.org/search/?type=Annotation&annotation_type=element+gene+regulatory+interaction+predictions&software_used.software.name=abc-enhancer-gene-prediction-encode_v1) or by running the ABC pipeline available on their [GitHub](https://github.com/broadinstitute/ABC-Enhancer-Gene-Prediction) acquire cell-type-specific gene-enhancer links. We provide a script [here](https://github.com/JasonLinjc/EPInformer/tree/main/data) for downloading ABC enhancer-gene links from ENCODE for K562 and GM12878 cells.

### Gene expression prediction
To predict the developmental and housekeeping enhancer activity in *Drosophila melanogaster* S2 cells for new DNA sequences, please run:
```
# Clone this repository
git clone https://github.com/JasonLinjc/EPInformer.git
cd EPInformer

# download the trained EPInformer model from zenodo ()

# create 'EPInformer_env' conda environment by running the following:
conda create --name EPInformer_env python=3.8 torch pandas scipy scikit-learn
source activate EPInformer_env
pip install pyranges pyfaidx kipoiseq
```

### Enhancer-gene links prediction

### Enhancer activity prediction and TF motif discovery
