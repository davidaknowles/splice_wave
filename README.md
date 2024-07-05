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
TemporalConvNet: this uses two paths of computation. One is causal/autoregressiv (i.e. uses causal convs) and processes the part being generated (i.e., isoform structure) while the other path is noncausal/bidirectional and processes whatever is being conditioned on (i.e., gene sequence). The noncausal path feeds into the causal at every layer, but not vis versa (or future info would contaminate the causal signal. 

ConvNet: just the noncausal/bidirectional part of TemporalConvNet. 

SpliceAI_10k: replication of the SpliceAI arch but flexible number of input and output channels. 

## Results so far: 
 - TemporalConvNet (TCN) with mamba added for the causal part doesn't do too well: can generate but doesn't match true isoform structure
 - Possibly the noncausal convnet is weak compared to spliceAI arch
 - Pure mambanet LM of seq+is_exon is hard to decode: i.e. fixing seq and trying to optimize is_exon just collapses to all intron or a mess. Could still use it to score a single variant. Could also try a DeepExplorationNet (DEN) approach to generate isoforms, but seems painful. 

BERT style MLM training, with SpliceAI arch. Isn't learning seq->is_exon well. 
 - with pure masking (no cheat/corrupt) the LM is worse with is_exon context than without :( (checkpoints_spliceAI_BERT)
 - but with 80/10/10 mask/cheat/corrupt the MLM benefits a lot from the is_exon context (test seq_loss down to ~0.2 from ~1.5). Note log(4)=1.4 so 1.5 is possibly worse than random (checkpoints_spliceAI_BERT_801010). Not sure if predicting seq->is_exon better than random? Def not well anyway.
 - Next is to add seq->is_exon as an additional task in each batch. This helps some but not a ton. 
 - SpliceAI 64 channel does a little better than Mamba on is_exon prediction, but a little worse on the MLM task.
 - 

## Diffusion models
 - Can do conditioning, e.g. The "RePaint" paper https://arxiv.org/abs/2201.09865 does it in a somewhat heuristic resampling approach. Would training time resampling be helpful? https://arxiv.org/pdf/2208.04202.pdf
 - Confusing Ganguli paper talks about conditioning: https://arxiv.org/pdf/1503.03585.pdf
 - Using https://github.com/lucidrains/denoising-diffusion-pytorch, seems cleaner than https://github.com/madaan/minimal-text-diffusion
 - BitDiffusion paper suggests should be able to just run Gaussian DM directly on {-1,+1} embedding. Doesn't seem to learn seq->is_exon relationship, but does generate "real" looking isoforms with values centered at {-1,+1}
 - Tried having the diffusion likelihood just have a Bernoulli/logistic likelihood (unlike what Diffusion-LM does) but this generates all 0s at least naively (without clipping predicted x_0 to [-1,1]). 
 - Diffusion-LM approach is to embed tokens into continuous space and then do diffusion there. Could do one hot or 2 channel encoding of seq plus one channel for is_exon. If embedding isn't learned isn't this the same as BitDiffusion? Trying with learned embedding: checkpoints_diffusion_binary_emb. Currently this does is_exon + 2bit = 3 channels. These are Conv1d embedded still to 3 channels (probably too few?) 
 - Binary (Latent) Diffusion: run a binary diffusion with Bernoulli noise. Trying this (April 1st): checkpoints_bld. Seems pretty useless (both with onehot and 2bit sequence rep)? 
 - https://beckham.nz/2022/07/11/d3pms.html

## To try/to do: 
 - BiMambaNet rather than SpliceAI arch for BERT MLM
 - Iterative generate isoform from MLM: sample most confident position, then condition, iterate. 
 - binary "latent" diffusion with fewer steps (their default is 40, i'm using 1000)
 - how does the DM U-net compare to SpliceAI arch?
 - if joint seq+is_exon doesn't work focus on is_exon->seq
 - is 3 channel rep better than 5 channel? 5 channel for BLD DM seems bad
 - structured diffusion models: https://github.com/google-research/google-research/blob/master/d3pm/text/diffusion.py
 - check if my BLD works on MNIST at least (might need making 2d?)
 - Currently running logit_diffusion cat, i.e. separately modeling is_exon and sequence. 
