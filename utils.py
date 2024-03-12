import numpy as np
import gzip 

nuc_arr = {'A':0,'C':1,'G':2,'T':3}

def one_hot(seq, dtype=np.float32):
    seq_len = len(seq)
    arr_rep = np.zeros((seq_len, len(nuc_arr)), dtype=dtype) 
    for i in range(seq_len):
        if seq[i] in nuc_arr:
            arr_rep[i,nuc_arr[seq[i]]] = 1 
    return arr_rep

def get_fasta(fasta_file, verbose = False):
    current_chrom=None
    dic={}
    with gzip.open(fasta_file) as f:
        for l in f:
            l=l.decode().strip()
            if l[0]==">":
                current_chrom=l[1:].strip()
                dic[current_chrom]=[]
                if verbose: print("Loading "+current_chrom)
            else:
                if not current_chrom is None:
                    dic[current_chrom].append( l.strip() )

    return( {k:"".join(v).upper() for k,v in dic.items() } )

REVERSER=str.maketrans("AGCT","TCGA")
    
def reverse_complement(seq):
    return seq.translate(REVERSER)[::-1]

def fetch_sequence(dic, fasta_id, start, end, strand = "+"):
    if not fasta_id in dic:
        return None
    seq =  dic[fasta_id][int(start):int(end)]

    return seq if strand=="+" else reverse_complement(seq)


