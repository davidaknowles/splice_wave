# Splice wave

The idea is to generate complete isoforms by modeling P(Y|X) where Y is the distribution over isoforms and X is the gene sequence. 

Y is modelled autoregressively using causal convolutions but conditional on X. This is somewhat inspired by the WaveNet papers for audio generation. 

