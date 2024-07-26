
import utils
import numpy as np
import torch
import torch.nn.functional as F
import gzip
import collections
from collections import defaultdict
from functools import partial
import os

import gtf_loader

import itertools

try: # import the cython version if possible (considerably faster) 
    from one_hot import one_hot
except ImportError as e: # otherwise use pure pyton
    from utils import one_hot

def repeat_iterator(iterator, times):
    # Duplicate the iterator
    iterators = itertools.tee(iterator, times)
    
    def generator():
        for it in iterators:
            yield from it
    
    return generator()


def get_mask(
    B, 
    T, 
    to_mask, 
    missing_rate = 0.15, 
    cheat_rate = 0.1, 
    corrupt_rate = 0.1, 
    min_span = 30, 
    max_span = 300, 
    mask_same = False
): 
    """
    Generates a mask for a given tensor `to_mask` to simulate missing data for masked language modeling.

    Parameters:
        B (int): Batch size.
        T (int): Length of each sequence.
        to_mask (torch.Tensor): Tensor to be masked.
        missing_rate (float, optional): The rate of missing tokens in the generated mask. Defaults to 0.15.
        cheat_rate (float, optional): The probability that the values of already masked tokens are retained. Defaults to 0.1.
        corrupt_rate (float, optional): The probability of corrupting already masked tokens. Defaults to 0.1.
        min_span (int, optional): Minimum span length for masked tokens. Defaults to 30.
        max_span (int, optional): Maximum span length for masked tokens. Defaults to 300.
        mask_same (bool, optional): When corrupting whether to use the same token at every position of the span. Set this to true for channels that have very high autocorrelation (e.g., exonic vs intronic). Defaults to False.

    Returns:
        torch.Tensor: Boolean mask tensor with dimensions (B, T) representing the masked positions.
    """
    device = to_mask.device
    bert_mask = torch.zeros( (B, T), dtype = bool, device = device)
    for b in range(B): 
        while bert_mask[b,:].float().mean() < missing_rate: 
            #print(bert_mask[b,:].float().mean(), end = "\r")  
            span = np.random.randint(min_span, max_span)
            start = np.random.randint(0, T-span)
            bert_mask[b, start:start + span] = True
            p = np.random.rand()
            if p > cheat_rate: 
                to_mask[b, :, start:start + span] = 0. 
            if (1.-p) < corrupt_rate: # 0ing already done
                channel_on = np.random.randint(0, to_mask.shape[1], size=1 if mask_same else span) 
                to_mask[b, channel_on, torch.arange(start, start+span)] = 1.
    return bert_mask

def get_mask_np(
    to_mask, # T x C
    missing_rate = 0.15, 
    cheat_rate = 0.1, 
    corrupt_rate = 0.1, 
    min_span = 30, 
    max_span = 300, 
    mask_same = False
): 
    T = to_mask.shape[0]
    bert_mask = np.zeros(T, dtype = bool)
    while bert_mask.mean() < missing_rate: 
        #print(bert_mask[b,:].float().mean(), end = "\r")  
        span = np.random.randint(min_span, max_span)
        start = np.random.randint(0, T-span)
        bert_mask[start:start + span] = True
        p = np.random.rand()
        if p > cheat_rate: 
            to_mask[start:start + span, :] = 0. 
        if (1.-p) < corrupt_rate: # 0ing already done
            channel_on = np.random.randint(0, to_mask.shape[1], size=1 if mask_same else span) 
            to_mask[np.arange(start, start+span), channel_on] = 1.
    return bert_mask

# Function to generate a boolean vector with autocorrelation
def generate_autocorrelated_mask(size, true_prob=0.15, autocorr_len=2):
    # Step 1: Generate random noise
    noise = torch.randn(size) * (torch.rand(size) < 0.5)
    
    # Step 2: Smooth the noise using a Gaussian filter
    noise = noise[None,None,:] # Add batch and channel dimensions
    smoothed_noise = F.conv1d(noise, weight=torch.ones(1,1,autocorr_len*2+1), padding=autocorr_len)
    smoothed_noise = smoothed_noise.squeeze()  # Remove batch and channel dimensions
    
    # Step 3: Normalize and threshold the smoothed noise
    threshold = torch.quantile(smoothed_noise, 1 - true_prob)

    return (smoothed_noise > threshold).numpy()

def get_mask_np_efficient(
    to_mask, # T x C
    missing_rate = 0.15, 
    cheat_rate = 0.1, 
    corrupt_rate = 0.1
): 
    T,C = to_mask.shape
    
    regular_mask = generate_autocorrelated_mask(
        T, 
        true_prob = missing_rate * (1. - cheat_rate - corrupt_rate))
    to_mask[regular_mask, :] = 0.
    
    cheat_mask = generate_autocorrelated_mask(
        T, 
        true_prob = missing_rate * cheat_rate)
    
    corrupt_mask = generate_autocorrelated_mask(
        T,
        true_prob = missing_rate * corrupt_rate)
    
    to_mask[corrupt_mask, :] = 0.
    channel_on = np.random.randint(0, 4, size=int(corrupt_mask.sum()))
    to_mask[corrupt_mask,channel_on] = 1.

    return regular_mask | cheat_mask | corrupt_mask

def get_tpms(fn): # "ENCFF191YXW.tsv.gz"
    tpms = {}
    first = True
    with gzip.open(fn) as f: 
        for l in f: 
            if first: 
                first=False
                continue
            ss = l.decode().strip().split()
            transcript = ss[0]
            tpm = float(ss[5])
            tpms[transcript] = tpm
    return(tpms)

# TODO: mask beyond transcript boundaries (unless want to do altAPA/TSS)
# Can do this using 2D sample weights
def get_generator(
    genome_fn, 
    gtf_fn, 
    tpm_fn=None, 
    to_one_hot = True, 
    verbose = False,
    down_sample_ratio = 1.,
    num_devices = 0, 
    device_id = 0
):

    print(f"get_generator num_devices {num_devices} device_id {device_id}") 
    
    tpms = get_tpms(tpm_fn) if tpm_fn else {}

    (exons, genes) = gtf_loader.get_exons(gtf_fn)

    genome = utils.get_fasta(genome_fn, verbose = verbose)

    def get_gene(chroms = None, receptive_field=0, min_len = 0, max_len = 10000):

        for yield_count,(gene,chrom_strand) in enumerate(genes.items()):
            
            if down_sample_ratio != 1.: 
                if np.random.rand() > down_sample_ratio: 
                    continue
            if num_devices > 0: 
                if yield_count % num_devices != device_id:
                    continue
            
            if chroms: 
                if not chrom_strand.chrom in chroms: 
                    continue
            gene_start = np.inf
            gene_end = 0
            for transcript,exons_here in exons[gene].items():
                for exon in exons_here: 
                    gene_start = min(gene_start, exon.start)
                    gene_end = max(gene_end, exon.end)
            transcripts = list(exons[gene].keys())
            num_transcripts = len(transcripts)
            
            if (gene_end - gene_start) > max_len: 
                gene_start = np.random.randint(gene_start, gene_end - max_len)
                gene_end = gene_start + max_len
            
            is_exon = np.zeros((num_transcripts, gene_end - gene_start), dtype=np.float32)

            for transcript_idx,transcript in enumerate(transcripts):
                exons_here = exons[gene][transcript]
                for exon in exons_here: 
                    exon_start = (exon.start-gene_start) if (chrom_strand.strand=="+") else (gene_end - exon.end)
                    exon_end = (exon.end-gene_start) if (chrom_strand.strand=="+") else (gene_end - exon.start)            
                    is_exon[transcript_idx, exon_start:exon_end] = 1

            offset = -1 if (chrom_strand.strand=="+") else 0
            seq = utils.fetch_sequence(genome, chrom_strand.chrom, gene_start + offset - receptive_field, gene_end + offset + receptive_field, chrom_strand.strand)
            # TODO: check seq is valid
            if seq is None: continue
            if len(seq) != (gene_end - gene_start + receptive_field * 2): continue
            start_di = [] # record dinucleotides sequences to check indexing is correct
            end_di = []
            weights = np.zeros(num_transcripts, dtype=np.float32)
            for transcript_idx,transcript in enumerate(transcripts): # could probably merge this loop with the one above
                weights[transcript_idx] = tpms.get(transcript,0.0)
                exons_here = exons[gene][transcript]
                for exon_idx,exon in enumerate(exons_here):
                    if exon_idx > 0 and exon_idx < len(exons_here)-1:
                        exon_start = (exon.start-gene_start) if (chrom_strand.strand=="+") else (gene_end - exon.end)
                        start_di.append( seq[exon_start-2:exon_start] )
                        exon_end = (exon.end-gene_start) if (chrom_strand.strand=="+") else (gene_end - exon.start)
                        end_di.append(seq[exon_end+1:exon_end+3])

            if len(start_di)==0: continue
            start_correct = np.mean( np.array(start_di) == "AG")
            end_correct = np.mean( np.array(end_di) == "GT")
            #assert(start_correct > .5)
            #assert(end_correct > .5)
            #print("Canonical start %f end %f (n=%s) strand %s " % (start_correct, end_correct, len(start_di), chrom_strand.strand))

            one_hot_enc = one_hot(seq) if to_one_hot else seq # one is length x 4
            #one_hot = np.tile(utils.one_hot(seq), (is_exon.shape[0],1,1))
            is_exon = is_exon[:,:,np.newaxis] # num_transcripts x length x 1
            #sample_weights = np.random.rand(is_exon.shape[0])
            # sample_weights = np.random.rand(is_exon.shape[0],is_exon.shape[1]) # requires sample_weight_mode="temporal" in model.compile
            #yield(((is_exon, one_hot), is_exon, sample_weights))
            #yield(((is_exon, one_hot), is_exon, weights)
            if min_len: 
                T = is_exon.shape[1]
                if T < min_len: 
                    pad = min_len - T 
                    is_exon = np.pad(
                        is_exon, 
                        ( (0,0), (0,pad), (0,0) )) 
                    
                    one_hot_enc = np.pad(
                        one_hot_enc, 
                        ( (0,pad), (0,0) ))
            
            yield(is_exon, one_hot_enc, weights)
            
    return(get_gene)

def collate_helper(x, min_len = 0, device = "cpu"): 
    if isinstance(x[0], (int, np.int32, np.int64)):
        return torch.LongTensor(x).to(device)
    elif isinstance(x[0], (float, np.float32, np.float64)):
        return torch.FloatTensor(x).to(device)
    elif isinstance(x[0], (np.ndarray, collections.abc.Sequence)):
        x = [ torch.tensor(g) for g in x ]
        padded_seq = torch.nn.utils.rnn.pad_sequence(x, batch_first = True, padding_value = 0.).to(device)
        T = padded_seq.shape[1]
        if T < min_len: 
            pad = (0, 0, 0, min_len - T)  # (pad_left, pad_right, pad_top, pad_bottom)
            padded_seq = F.pad(padded_seq, pad)
        lengths = torch.tensor([ len(g) for g in x ])
        return (padded_seq, lengths)
    else: 
        raise ValueError(f"Don't know how to collate {type(x[0])}")

def custom_collate(batch, min_len = 0, device = "cpu"):
    return [ collate_helper(g, min_len = min_len, device = device) for g in zip(*batch) ]

class TranscriptDataset(torch.utils.data.IterableDataset):

    def __init__(self, get_gene, chroms, receptive_field, max_len, min_len = 0, repeats = 1):
        super().__init__()
        
        self.get_gene = get_gene
        self.chroms = chroms
        self.receptive_field = receptive_field
        self.min_len = min_len
        self.max_len = max_len

        self.repeats = repeats
    
        chars = "ACGT"
        stoi = defaultdict(lambda: 4, {ch:i for i,ch in enumerate(chars)})
        itos = defaultdict(lambda: "N", {i:ch for i,ch in enumerate(chars)})
        
        self.encode = lambda xx: np.array([stoi[x] for x in xx], dtype = np.int64)
        self.decode = lambda xx: ''.join([itos[x] for x in xx])

    def __iter__(self):

        my_it = self.get_gene(self.chroms, self.receptive_field, min_len = self.min_len, max_len = self.max_len)
        repeated_it = repeat_iterator(my_it, self.repeats)
        
        for (is_exon, seq, weights) in repeated_it: 
            if weights.sum() == 0: continue
            to_keep = np.where(weights)[0]
            is_exon = is_exon[to_keep,:,:]
            weights = weights[to_keep] # won't be used for now, might sample later
            weights = weights / weights.sum()
    
            to_keep = np.argmax(weights)
            is_exon = is_exon[to_keep,:,:]
            weights = weights[to_keep]
    
            seq_enc = seq if isinstance(seq, np.ndarray) else self.encode(seq)

            one_hot_masked = seq_enc.copy()
            seq_mask = get_mask_np_efficient(one_hot_masked)

            yield is_exon, seq_enc, one_hot_masked, seq_mask, weights

def get_dataloader(
    get_gene, 
    chroms, 
    receptive_field = 0, 
    batch_size = 10, 
    min_len = 0, 
    max_len = 30000, 
    device = "cpu", 
    num_workers = 0,
    repeats = 1): 

    collate_fn = partial( custom_collate, min_len = min_len, device = device)
    
    dataset = TranscriptDataset(
        get_gene, 
        chroms,
        receptive_field, 
        min_len = min_len,
        max_len = max_len,
        repeats = repeats
    )

    return torch.utils.data.DataLoader(
        dataset, 
        collate_fn = collate_fn,
        batch_size = batch_size, 
        num_workers = num_workers
    )

