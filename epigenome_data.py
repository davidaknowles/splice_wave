import utils
import one_hot
import numpy as np
import torch
import torch.nn.functional as F
import gzip
import re
import time
import collections
from collections import defaultdict
from functools import partial
import os
import pandas as pd
from pathlib import Path
from utils import get_fasta
import wget

# add species to genome_urls
vertebrate_epigenomes = Path("vertebrate_epigenomes")

genome_urls = pd.read_csv(vertebrate_epigenomes / "genome_urls.tsv", sep = "\t")
genomes_dir = Path("genomes")
genomes_dir.mkdir(parents=True, exist_ok=True)

for i in range(genome_urls.shape[0]): 
    genome = genome_urls.loc[i, "genome"]
    species = genome_urls.loc[i, "species"]
    url = genome_urls.loc[i, "url"]
    gfile = genomes_dir / f"{genome}.fa.gz"
    if not gfile.exists():
        print(f"Downloading {genome}") 
        wget.download(url, out = str(gfile))

    print(f"Loading {genome}") 

    start_time = time.time()
    genome_dict = get_fasta(gfile, verbose = True)
    print(time.time() - start_time)

    beddir = vertebrate_epigenomes / f"{species}_{genome}"

    files = beddir.glob("*.bed.gz")
    for f in files: 
        print(f)
        bed = pd.read_csv(
            f, 
            sep="\t", 
            usecols = [0,1,2],
            names = ("chrom", "start", "end"),
            low_memory=False
        )
        bed_chroms = bed.chrom.unique() # e.g. "chr1" in mouse

        # remove .1 or .2 from end
        bed_chroms = [ re.sub(r'\.\d+$', '', g) for g in bed_chroms ]
        genome_chroms = [ re.sub(r'\.\d+$', '', g) for g in genome_dict.keys() ]

        bed_chroms = [ re.sub(r'^[Cc]hrUn_', '', g) for g in bed_chroms ]
        genome_chroms = [ re.sub(r'^[Cc]hrUn_', '', g) for g in genome_chroms ]

        # remove "chr" or "Chr" from start
        bed_chroms = [ re.sub(r'^[cC]hr', '', g) for g in bed_chroms ]
        genome_chroms = [ re.sub(r'^[Cc]hr', '', g) for g in genome_chroms ]

        bed_chroms = [ re.sub(r'^0', '', g) for g in bed_chroms ] # fix 01 to 1 etc

        bed_chroms = [ g for g in bed_chroms if not "AEMK" in g ]
        bed_chroms = [ g for g in bed_chroms if not "FPKY" in g ]
        # genome_dict has e.g. "1" in mouse. i think gencode would match better? 

        if genome == "Rnor_6.0": 
            bed_chroms = [ re.sub(r'v1$', '', g) for g in bed_chroms ] # maybe can do this with all? 

        bed_chroms = [ g for g in bed_chroms if not "scaffold" in g ]
        bed_chroms = [ g for g in bed_chroms if not "unplaced" in g ]
        bed_chroms = [ g for g in bed_chroms if not "NW_0201" in g ]
        bed_chroms = [ g for g in bed_chroms if not "_random" in g ]
        
        # GRCm38
        # change url to https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M10/GRCm38.primary_assembly.genome.fa.gz

        # bed files aren't consistent for pig :( 
        pd.Series(bed_chroms)[~pd.Series(bed_chroms).isin(genome_chroms)]
        assert pd.Series(bed_chroms).isin(genome_chroms).mean() > 0.9 

def clean_chroms(chroms): 
    return [ re.sub(r'\.\d+$|^[Cc]hrUn_|^[cC]hr|v1$|^0', '', g) for g in chroms ]

all_meta = pd.read_csv(vertebrate_epigenomes / "all_meta.tsv", sep="\t")
all_bed = []

for i in range(genome_urls.shape[0]): 
    genome = genome_urls.loc[i, "genome"]
    species = genome_urls.loc[i, "species"]
    url = genome_urls.loc[i, "url"]
    print(genome,species)
    build_bedfile = vertebrate_epigenomes / f"{species}_{genome}" / "cache.parquet" 
    if build_bedfile.exists(): 
        all_bed.append(pd.read_parquet(build_bedfile))
        continue
    gfile = genomes_dir / f"{genome}.fa.gz"
    if not gfile.exists():
        print(f"Downloading {genome}") 
        wget.download(url, out = str(gfile))

    print(f"Loading {genome}") 

    start_time = time.time()
    genome_dict = get_fasta(gfile, verbose = True)
    print(time.time() - start_time)

    genome_chroms = clean_chroms(genome_dict.keys())
    
    current_meta = all_meta[all_meta["genome"]==genome]
    
    beds = []
    for i in range(len(current_meta)): 
        bed = pd.read_csv(
            vertebrate_epigenomes / current_meta.iloc[i]["final"], 
            sep="\t", 
            usecols = [0,1,2],
            names = ("chrom", "start", "end"),
            low_memory=False
        )
        bed["tissue"] = current_meta.iloc[i]["tissue"]
        bed["assay"] = current_meta.iloc[i]["assay"]
        bed["species"] = species # just training on one genome at a time? 
        bed["genome"] = genome
        beds.append(bed)
    
    bed = pd.concat(beds, axis = 0)
    
    bed.chrom = clean_chroms(bed.chrom)
    known_chrom = bed.chrom.isin(genome_chroms)
    print("Prop known chroms",known_chrom.mean())
    assert known_chrom.mean() > 0.90
    bed = bed[known_chrom]
    bed.to_parquet(build_bedfile)
    all_bed.append(bed)

for i in range(len(all_bed)): 
    all_bed[i]["start"] = all_bed[i]["start"].astype("int64")
    all_bed[i]["end"] = all_bed[i]["end"].astype("int64")

all_bed_concat = pd.concat(all_bed, axis = 0)
all_bed_concat.to_parquet(vertebrate_epigenomes / "vertebrate_epigenomes.parquet")



[ b.start.dtype == 'int64' for b in all_bed ]
