# Download K562 training data from Zenodo
!wget https://zenodo.org/records/12738705/files/K562_DNase_ENCFF257HEE_2kb_4DNFITUOMFUQ_enhancer_promoter_encoding.h5.zip -P ./data/
!wget https://zenodo.org/records/12738705/files/K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated.zip -P ./data/
!unzip -o -qq ./data/K562_DNase_ENCFF257HEE_2kb_4DNFITUOMFUQ_enhancer_promoter_encoding.h5.zip -d ./data/ -o
!unzip -o -qq ./data/K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated.zip -d ./data/ -o

# Download GM2878 training data from Zenodo
!wget https://zenodo.org/records/12738705/files/GM12878_DNase_ENCFF020WZB_2kb_4DNFI1UEG1HD_promoter_enhancer_encoding.h5.zip -P ./data/
!wget https://zenodo.org/records/12738705/files/GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated.zip -P ./data/
!unzip -o -qq ./data/GM12878_DNase_ENCFF020WZB_2kb_4DNFI1UEG1HD_promoter_enhancer_encoding.h5.zip -d ./data/ -o
!unzip -o -qq ./data/GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated.zip -d ./data/ -o
