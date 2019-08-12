import gzip
import sys
from collections import namedtuple

Interval = namedtuple("Interval", ("start","end"))

ChromStrand = namedtuple("ChromStrand", ("chrom","strand"))

def my_open(filename, mode):
    return( gzip.open(filename,mode) if (filename[-2:] == "gz") else open(filename,mode) )

def get_exons(gtf_filename):
    exons={} # gene -> transcript -> exons(list of Interval)
    genes={} # gene -> ChromStrand
    with my_open(gtf_filename,"r") as f:
        for l in f:
            l = l.decode()
            if l[0] == "#": continue
            (chrom, _, feature, start, end, _, strand, _, the_rest) = l.strip().split("\t")
            if not feature in ("exon","five_prime_utr","three_prime_utr"): continue
            #if not feature in ("exon"): continue
            try:
                meta = dict( [ g.strip().split(' ') for g in the_rest.split("; ") ]  )
            except ValueError as e: 
                print(l)
                raise(e)
            
            meta = { k:v.strip('"') for k,v in meta.items() }

            exon=Interval(start=int(start), end=int(end))
            if exon.start==exon.end: continue # length 0 UTR

            transcript_id = meta["transcript_id"]
            gene_id = meta["gene_id"]

            if not gene_id in exons: exons[gene_id]={}
            if not transcript_id in exons[gene_id]: exons[gene_id][transcript_id] = []
            exons[gene_id][transcript_id].append(exon)

            if not gene_id in genes: genes[gene_id] = ChromStrand(chrom = chrom, strand = strand)
            
    return(exons, genes)
