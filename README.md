# Splice wave

The idea is to generate complete isoforms by modeling P(Y|X) where Y is the distribution over isoforms and X is the gene sequence. 

Y is modelled autoregressively using causal convolutions but conditional on X. This is somewhat inspired by the WaveNet papers for audio generation. 

## SSM version with Mamba

Annoyingly convolutions expect
batch x channels x time
whereas Mamba expects
batch x time x channels
which is admittedly nicer when used with nn.Linear (just operates on the last channel dim). 

## Models
TemporalConvNet: this uses two paths of computation. One is causal/autoregressiv (i.e. uses causal convs) and processes the part being generated (i.e., isoform structure) while the other path is noncausal/bidirectional and processes whatever is being conditioned on (i.e., gene sequence). The causal path feeds into the noncausal at every layer, but not vis versa (or future info would contaminate the causal signal. 

ConvNet: just the noncausal/bidirectional part of TemporalConvNet. 

SpliceAI_10k: replication of the SpliceAI arch but flexible number of input and output channels. 

## Results so far: 
TCN with mamba added for the causal part doesn't do too well: can generate but doesn't match true isoform structure
Possibly the noncausal convnet is weak compared to spliceAI arch
Pure mambanet LM of seq+is_exon is hard to decode: i.e. fixing seq and trying to optimize is_exon just collapses to all intron or a mess. Could still use it to score a single variant. Could also try a DEN approach to generate isoforms, but seems painful. 

BERT style MLM training, with SpliceAI arch. Isn't learning seq->is_exon well. 
 - with pure masking (no cheat/corrupt) the LM is worse with is_exon context than without :( (checkpoints_spliceAI_BERT)
 - but with 80/10/10 mask/cheat/corrupt the MLM benefits a lot from the is_exon context (test seq_loss down to ~0.2 from ~1.5). Note log(4)=1.4 so 1.5 is possibly worse than random (checkpoints_spliceAI_BERT_801010). Not sure if predicting seq->is_exon better than random? Def not well anyway.
 - Next is to add seq->is_exon as an additional task in each batch. 

## To try/to do: 
BiMambaNet rather than SpliceAI arch for BERT MLM
Iterative generate isoform from MLM: sample most confident position, then condition, iterate. 
Diffusion model. Can this do conditioning? Was hoping yes through the Gaussian? 
 - https://github.com/madaan/minimal-text-diffusion
 - Confusing Ganguli paper talks about conditioning: https://arxiv.org/pdf/1503.03585.pdf
 - The "RePaint" paper https://arxiv.org/abs/2201.09865 does it in a somewhat heuristic resampling approach
 - Nice think would be you can train the model then try diff conditioning approaches

