from bs4 import BeautifulSoup
import requests
from pathlib import Path
import pandas as pd

def download_file(url, local_filename):
    # Stream the response to avoid loading the entire file into memory
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Check for HTTP errors
        with open(local_filename, 'wb') as f:
            # Iterate over the response content in chunks
            for chunk in response.iter_content(chunk_size=8192): 
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
    return local_filename

def geo_downloader(gse, href_filter, outputdir = None): 

    if outputdir is None: 
        outputdir = Path(gse)
        outputdir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/suppl/"
    
    response = requests.get(base_url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve data for {gse_id}")
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all links to supplementary files
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href_filter(href):
            print("Downloading",href)
            download_file(base_url + href, outputdir / href)
        else: 
            print("not downloading",href)

geo_downloader("GSE195592", lambda href: href.endswith("bw")) # bw_to_peaks.sh has commands to convert to bedgraph then macs2 peaks
geo_downloader("GSE242357", lambda href: href.endswith("txt.gz"))
geo_downloader("GSE164361", lambda href: href.endswith("bed.gz"))

# these are corrupted
# a = pd.read_csv("GSE242357/GSE242357_motrpac_pass1b-06_t59-kidney_epigen-atac-seq_counts.txt.gz", sep = "\t", low_memory = False) # fails 

