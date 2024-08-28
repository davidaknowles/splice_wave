import pandas as pd
from pathlib import Path
import wget
import concurrent.futures
import pyarrow as pa
import pyarrow.parquet as pq
from utils import get_fasta

# add species to genome_urls
vertebrate_epigenomes = Path("vertebrate_epigenomes")

genome_urls = pd.read_csv(vertebrate_epigenomes / "genome_urls.tsv", sep = "\t")
genomes_dir = Path("genomes")
genomes_dir.mkdir(parents=True, exist_ok=True)

# Define the download function
def download_genome(data):
    genome, species, url, genomes_dir = data
    gfile = genomes_dir / f"{genome}.fa.gz"
    if not gfile.exists():
        print(f"Downloading {genome}")
        wget.download(url, out=str(gfile))

    genome_dict = get_fasta(gfile, verbose = False)
    sequence_id_array = pa.array(genome_dict.keys())
    sequence_array = pa.array(genome_dict.values())

    table = pa.Table.from_arrays([sequence_id_array, sequence_array], names=['sequence_id', 'sequence'])

    pq.write_table(table, genomes_dir / f"{genome}.parquet") 

# Create a list of tuples for the download function
download_data = [(row['genome'], row['species'], row['url'], genomes_dir) for _, row in genome_urls.iterrows()]

# Download files in parallel using multiprocessing
with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(download_genome, download_data)

print("Downloads complete.")
