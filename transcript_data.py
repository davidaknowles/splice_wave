
import utils
import numpy as np

import gzip

import gtf_loader

#%%
(exons, genes) = gtf_loader.get_exons("gencode.v24.annotation.gtf.gz")

genome = utils.get_fasta("hg38.fa.gz")

#%%
tpms = {}
first = True
with gzip.open("ENCFF191YXW.tsv.gz") as f: 
    for l in f: 
        if first: 
            first=False
            continue
        ss = l.decode().strip().split()
        transcript = ss[0]
        tpm = float(ss[5])
        tpms[transcript] = tpm

# TODO: mask beyond transcript boundaries (unless want to do altAPA/TSS)
# Can do this using 2D sample weights
#%%
def get_gene(receptive_field=0):
    for gene,chrom_strand in genes.items():
        gene_start = np.inf
        gene_end = 0
        for transcript,exons_here in exons[gene].items():
            for exon in exons_here: 
                gene_start = min(gene_start, exon.start)
                gene_end = max(gene_end, exon.end)
        transcripts = list(exons[gene].keys())
        num_transcripts = len(transcripts)
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
        start_di = []
        end_di = []
        for transcript_idx,transcript in enumerate(transcripts):
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
        
        one_hot = utils.one_hot(seq)
        #one_hot = np.tile(utils.one_hot(seq), (is_exon.shape[0],1,1))
        is_exon = is_exon[:,:,np.newaxis]
        #sample_weights = np.random.rand(is_exon.shape[0])
        # sample_weights = np.random.rand(is_exon.shape[0],is_exon.shape[1]) # requires sample_weight_mode="temporal" in model.compile
        #yield(((is_exon, one_hot), is_exon, sample_weights))
        yield(((is_exon, one_hot), is_exon))

#%%
