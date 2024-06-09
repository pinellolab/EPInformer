<p align="center">
  <img height="500" src="images/EPInformer.png">
</p>

Welcome to the EPInformer framework repository! EPInformer is a scalable deep learning framework for gene expression prediction by integrating promoter-enhancer sequences with epigenomic signals. EPInformer is designed for three key applications: 1) Accurately predicting gene expression levels using promoter-enhancer sequences, epigenomic signals, and chromatin contacts; 2) Efficiently identifying cell-type-specific enhancer-gene interactions, validated by CRISPR perturbation experiments; 3) Precisely predicting enhancer activity and identifying transcription factor binding motifs from sequences.

This repository can be used to run the EPInformer model and get gene expression predictions and prioritize enhancer-gene interactions for input sequences and epigenomic signals.

We also provide information and instructions for how to train different versions of EPInformer given diffenet inputs including DNA sequence, epigemoic signals and chomatine contacts.

### Requirements

EPInformer requires Python 3.6+ and Python packages PyTorch (>=2.1). You can follow PyTorch installation steps [here](https://pytorch.org/get-started/locally/).

### Setup

EPInformer requires ABC enhancer-gene data for training and prediting gene expression. Please download the ABC data from [ENCODE](https://www.encodeproject.org/search/?type=Annotation&annotation_type=element+gene+regulatory+interaction+predictions&software_used.software.name=abc-enhancer-gene-prediction-encode_v1) or running the pipeline from their [github](https://github.com/broadinstitute/ABC-Enhancer-Gene-Prediction). We provide the script for downloading ABC enhancer-gene links of K562 and GM12878 [here](https://github.com/JasonLinjc/EPInformer/tree/main/data).

### Gene expression prediction

