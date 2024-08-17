import pandas as pd
from pathlib import Path
import wget
import concurrent.futures

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

# Create a list of tuples for the download function
download_data = [(row['genome'], row['species'], row['url'], genomes_dir) for _, row in genome_urls.iterrows()]

# Download files in parallel using multiprocessing
with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(download_genome, download_data)

print("Downloads complete.")
