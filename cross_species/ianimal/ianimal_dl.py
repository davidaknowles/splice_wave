import requests
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def merge_intervals(df):
    
    df.sort_values(by=['chrom', 'start'], inplace=True)
    
    merged = []

    for index, row in df.iterrows():
        # If merged list is empty, add the first interval
        if not merged:
            merged.append((row['chrom'], row['start'], row['end']))
        else:
            last = merged[-1]

            # Check if current interval overlaps or is adjacent to the last merged interval
            if row['chrom'] == last[0] and row['start'] <= last[2]:
                merged[-1] = (last[0], last[1], max(last[2], row['end']))
            else:
                merged.append((row['chrom'], row['start'], row['end']))

    return pd.DataFrame(merged, columns=['chrom', 'start', 'end'])

def get_sample_meta(species_id, sample_name): 
    url = f"https://ianimal.pro/api/v1/epigenome/sample/{species_id}/{sample_name}"
    response = requests.get(
        url = url, 
        params = {"skip": 0, "limit": 100},
        headers={ "accept": "application/json" })

    if response.status_code != 200: 
        print(f"Warning: couldn't retrieve {url}")
        return None

    data = response.json()

    return { k:data[k] for k in ['targat', 'tissue ontology', 'tissue', 'peak number', 'peak length'] }

if __name__ == "__main__":

    response = requests.get(
        url = "https://ianimal.pro/api/v1/species/", 
        params = {"skip": 0, "limit": 100},
        headers={ "accept": "application/json" })

    data = response.json()

    data_i = int(sys.argv[1]) # this job will process this species id
    species_id = data[data_i]['id']
    species_name = data[data_i]['full_name'].replace(' ', '_')

    # make a dir to store data for this species
    species_dir = Path("data") / species_name
    species_dir.mkdir(exist_ok = True)

    # this file contains all the data for the species, but the sample_id is not informative
    # we will use get_sample_meta() to look up the tissue+assay for each sample
    filename = species_dir / f"{species_name}_peak_list.gz"

    response = requests.get(
        url = f"https://direct.ianimal.pro/gz/Omics_Data/{species_name}/{species_name}_peak_list.gz")

    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File '{filename}' downloaded successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
        species_dir.rmdir()

    peaks = pd.read_csv(filename, sep = "\t", dtype={'chrom': str})

    # only keep confident peaks
    peaks = peaks[ peaks["-log10qvalue"] > -np.log10(0.01) ]

    # filter excessively long or short peaks
    peak_len = peaks.end - peaks.start
    peaks = peaks[ (peak_len <= 5000) & (peak_len >= 50) ]

    # remove samples with <10000 confident peaks
    peak_counts = peaks.sample_id.value_counts()
    to_keep = peak_counts.index[peak_counts >= 1e4]
    peaks = peaks[peaks.sample_id.isin(to_keep)]

    # get tissue & assay info for every sample id
    sample_meta = pd.DataFrame(
        [ get_sample_meta(species_id, sample_name) for sample_name in peak_counts.index ], 
        index = peak_counts.index)

    sample_meta.rename(columns={'targat': 'target'}, inplace=True) # correct their typo

    # for each unique (assay, tissue) combo, merge all files 
    target_tissue_combos = sample_meta[ ["target", "tissue"] ].drop_duplicates() 
    for row in target_tissue_combos.iterrows(): 
        try: 
            print(row)
            row = row[1]
            to_merge = sample_meta[(sample_meta.tissue == row.tissue) & (sample_meta.target == row.target)]
            peaks_to_merge = peaks[peaks.sample_id.isin(to_merge.index)]
            if len(peaks_to_merge) < 10000: 
                continue
            merged = merge_intervals(peaks_to_merge)
            merged.to_csv(species_dir / f"{row.tissue}_{row.target}.bed.gz", sep='\t', index=False, header=False)
        except: 
            print(f"Warning: failed to merge tissue {row.tissue} target {row.target} for {species_name}")
