#!/bin/sh

wget https://www.encodeproject.org/files/ENCFF191YXW/@@download/ENCFF191YXW.tsv
gzip ENCFF191YXW.tsv
#wget http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_24/gencode.v24.annotation.gtf.gz