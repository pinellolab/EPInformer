# Download K562 training data from Zenodo
wget https://zenodo.org/records/13233337/files/K562_200CREs-gene_RPM_4feats.hdf5.gz -P ./data/
wget https://zenodo.org/records/13233337/files/K562_ABC_EGLinks.zip -P ./data/
gunzip -o -qq ./data/K562_200CREs-gene_RPM_4feats.hdf5.gz -d ./data/
unzip -o -qq ./data/K562_ABC_EGLinks.zip -d ./data/
