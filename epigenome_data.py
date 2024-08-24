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
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
import transcript_data
import requests

def clean_chroms(chroms): 
    return [ re.sub(r'\.\d+$|^[Cc]hrUn_|^[cC]hr|v1$|^0', '', g) for g in chroms ]

def parquet_to_genome_dict(fn): 
    table = pq.read_table(fn) 
    chrom_names = clean_chroms(table['sequence_id'].to_pylist())
    return dict(zip(chrom_names, table['sequence'].to_pylist()))

class BedDataset(torch.utils.data.Dataset):

    def __init__(self, data, genome_dicts, width=1000, mask = False):

        super().__init__()
        self.width = width
        assert width % 2 == 0
        self.data = data

        self.tissue_int = data['tissue'].cat.codes.reset_index(drop=True)
        self.species_int = data['species'].cat.codes.reset_index(drop=True)
        self.assay_int = data['assay'].cat.codes.reset_index(drop=True)
        
        self.genome_dicts = genome_dicts

        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        peak_start = self.data["start"].iloc[i]
        peak_end = self.data["end"].iloc[i]
        genome = self.data["genome"].iloc[i]
        chrom = self.data["chrom"].iloc[i]
        
        genome_dict = self.genome_dicts[ genome ]

        peak_width = peak_end - peak_start
        
        if peak_width > self.width: 
            peak_start = np.random.randint(peak_start, peak_end - self.width)
            peak_end = peak_start + self.width
        elif peak_width < self.width: # could pad instead? 
            mid = int(0.5*(peak_start + peak_end))
            peak_start = mid - self.width // 2
            peak_end = mid + self.width // 2
            if peak_start < 0: 
                peak_start = 0
                peak_end = self.width
            if peak_end > len(genome_dict[chrom]): 
                peak_end = len(genome_dict[chrom])
                peak_start = peak_end - self.width
        
        seq = utils.fetch_sequence(
            genome_dict, 
            chrom, 
            peak_start, 
            peak_end, 
            "+" if (np.random.rand() < 0.5) else "-" # should return RC with 50% chance
        )

        #assert not seq is None
        #assert len(seq) == self.width

        if seq is None: 
            print("Warning: seq is None")
            one_hot_enc = np.zeros((self.width, 4))
        else: 
            # will be L x 4
            one_hot_enc = one_hot.one_hot(seq) # if to_one_hot else seq
            if len(seq) < self.width: 
                rows_to_pad = self.width - len(seq)
                one_hot_enc = np.pad(one_hot_enc, ((0, rows_to_pad), (0, 0)))

        to_return = [self.species_int[i], self.tissue_int[i], self.assay_int[i], one_hot_enc]

        if self.mask: 
            one_hot_masked = one_hot_enc.clone()
            mask = transcript_data.get_mask_np_efficient(one_hot_enc) 
            to_return += [ one_hot_masked, mask ]
        
        return to_return

def load_data(genome_subset = None, width = 1000): 
    vertebrate_epigenomes = Path("vertebrate_epigenomes")

    genome_urls = pd.read_csv(vertebrate_epigenomes / "genome_urls.tsv", sep = "\t")
    genomes_dir = Path("genomes")
    genomes_dir.mkdir(parents=True, exist_ok=True)
    
    genome_dict = {}
    for i in range(genome_urls.shape[0]): 
        genome = genome_urls.loc[i, "genome"]
        if genome_subset is not None: 
            if not genome in genome_subset: 
                continue
        species = genome_urls.loc[i, "species"]
        print(genome)
        parquet_genome = genomes_dir / f"{genome}.parquet"
        genome_dict[genome] = parquet_to_genome_dict(parquet_genome)
    
    bed_data = pd.read_parquet( vertebrate_epigenomes / "vertebrate_epigenomes.parquet")

    if genome_subset is not None: 
        bed_data = bed_data[ bed_data["genome"].isin(genome_subset) ]
    
    bed_data['tissue'] = bed_data['tissue'].fillna('None').astype("category")
    bed_data['species'] = bed_data['species'].astype("category")
    bed_data['assay'] = bed_data['assay'].astype("category")

    return bed_data, genome_dict

if __name__ == "__main__":

    bed_data, genome_dict = load_data()

