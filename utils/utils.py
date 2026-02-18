# pure python backup for one_hot and get_fasta, in case Cython is not available.
import numpy as np
import gzip 
import time

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
                current_chrom=l[1:].split()[0].strip()
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


class RateTracker(object):

  def __init__(self, smooth_factor=0.4):
    self._smooth_factor = smooth_factor
    self._start_time = time.time()
    self._partial_time = self._start_time
    self._partial_count = 0.0
    self._partial_rate = None
    self._count = 0.0

  def _update(self, now, rate):
    self._partial_count += self._count
    self._count = 0.0
    self._partial_time = now
    self._partial_rate = rate

  def add(self, count):
    self._count += count

  def _smooth(self, current_rate):
    if self._partial_rate is None:
      smoothed_rate = current_rate
    else:
      smoothed_rate = ((1 - self._smooth_factor) * current_rate +
                       self._smooth_factor * self._partial_rate)
    return smoothed_rate

  def rate(self):
    now = time.time()
    delta = now - self._partial_time
    report_rate = 0.0
    if delta > 0:
      report_rate = self._smooth(self._count / delta)
      self._update(now, report_rate)
    return report_rate

  def global_rate(self):
    delta = time.time() - self._start_time
    count = self._partial_count + self._count
    return count / delta if delta > 0 else 0.0



