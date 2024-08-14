#!/bin/sh

wget https://www.encodeproject.org/files/ENCFF191YXW/@@download/ENCFF191YXW.tsv
gzip ENCFF191YXW.tsv
#wget http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_24/gencode.v24.annotation.gtf.gz

# Collection of H3K27ac and ATAC from various primate brain regions
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130871/suppl/GSE130871_RAW.tar
tar -xf GSE130871_RAW.tar --wildcards '*narrowPeak.gz'
rm GSE130871_RAW.tar

# Zebrafish dev altas
# wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243256/suppl/GSE243256%5FZEPA.All.sample.bed.gz # this is per cell?! 
#wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243256/suppl/GSE243256_RAW.tar
#tar -xf GSE243256_RAW.tar --wildcards '*.bw'
get_atac.py
atac/bw_to_peaks.sh

# get human and mouse atac from ENCODE (Enformer?) 

# Some chicken and possum samples: GSE185775

# GSE195592 chicken and mouse (hair and feathers!) 
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE195nnn/GSE195592/suppl/GSE195592_RAW.tar
tar -xf GSE195592_RAW.tar --wildcards '*ATAC*narrowPeak.gz'
tar -xf GSE195592_RAW.tar --wildcards '*H3K27ac*narrowPeak.gz'
rm GSE130871_RAW.tar

# Chicken/mouse limb bud
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164738/suppl/GSE164738_RAW.tar
tar -xf GSE164738_RAW.tar --wildcards '*ATAC*bw'

# GSE158430 pig (P files), cow (M08 and M22), chicken (A, B files) multiple tissues, nice! 
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE158nnn/GSE158430/suppl/GSE158430_RAW.tar
tar -xf GSE158430_RAW.tar --wildcards '*ATAC*bed.gz'
tar -xf GSE158430_RAW.tar --wildcards '*H3K27ac*bed.gz'
rm GSE158430_RAW.tar

# Pig multiple tissues
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE143nnn/GSE143288/suppl/GSE143288_RAW.tar
tar -xf GSE143288_RAW.tar --wildcards '*.narrowPeak.gz'
# tar -xf GSE143288_RAW.tar --wildcards '*ATAC.narrowPeak.gz'

# GSE145619 frog! 
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE145nnn/GSE145619/suppl/GSE145619_RAW.tar
tar -xf GSE145619_RAW.tar --wildcards '*peaks.bed.gz'
