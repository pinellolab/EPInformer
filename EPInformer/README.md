All proposed EPInformer models in our study are based on this core architecture. EPInformer-promoter masks all enhancer sequences with a zero vector and does not use any extra features; EPInformer-PE is a sequence-only model, using distance as the only additional feature. EPInformer-PE-Activity employs pre-trained residual convolution layers and takes promoter-enhancer sequences, distance, enhancer activity, promoter activity, and mRNA half-life features as inputs; EPInformer-PE-Activity-HiC includes the additional HiC contact feature compared to EPInformer-PE-Activity; MHA for multi-head attention.

<p align="center">
  <img height="600" src="../images/detailed_EPInformer.png">
</p>


