from typing import cast
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
from functools import partial
from jax.scipy.special import logsumexp
import eqx_modules
import eqx_transformer
import gbst_jax
import equinox.nn as nn


class Charformer(eqx.Module): 

    tokenizer: gbst_jax.GBST
    transformer_stack: eqx_transformer.TransformerStack
    up: nn.ConvTranspose1d
    final: nn.Conv1d
    
    def __init__(
        self, 
        input_dim, 
        d_model, 
        output_dim,
        downsample_factor, 
        n_heads, 
        d_ff, 
        key, 
        max_block_size = None, 
        blocks = None, 
        num_layers = 3, 
        final_kernel_size = 7
    ): 
        super().__init__()
        keys = jr.split(key, 4)

        self.tokenizer = gbst_jax.GBST(
            input_dim = input_dim,
            d_model = d_model,                    # dimension of token and intra-block positional embedding
            max_block_size = max_block_size,           # maximum block size
            blocks = blocks,
            downsample_factor = downsample_factor,        # the final downsample factor by which the sequence length will decrease by
            key = keys[0]
        )
        self.transformer_stack = eqx_transformer.TransformerStack(
            num_layers = num_layers, 
            d_model = d_model, 
            n_heads = n_heads, 
            d_ff = d_ff,
            key = keys[1]
        )

        self.up = nn.ConvTranspose1d(d_model, d_model, kernel_size = downsample_factor, stride = downsample_factor, padding = "same", key = keys[2]) 

        self.final = nn.Conv1d(d_model, output_dim, final_kernel_size, padding = "same", key = keys[3])

    def __call__(self, x):
        _, L = x.shape
        char_x, x = self.tokenizer(x)
        x = self.transformer_stack(x)
        x = x.transpose() # L x D -> D x L
        x = self.up(x)[:,:L] + char_x.transpose() # is this guaranteed to come out length L or greater? 
        x = jax.nn.relu(x) 
        x = self.final(x) 
        return x


class Convformer(eqx.Module): 

    tokenizer: eqx_modules.Conv1DLayer
    transformer_stack: eqx_transformer.TransformerStack
    up: nn.ConvTranspose1d
    final: nn.Conv1d
    downsample_factor: int 
    causal: bool
    pooler: nn.AvgPool1d
    
    def __init__(
        self, 
        input_dim, 
        d_model, 
        output_dim,
        downsample_factor, 
        n_heads, 
        d_ff, 
        kernel_size, 
        key, 
        causal = False, 
        gated = False,
        num_layers = 3, 
        final_kernel_size = 7
    ): 
        super().__init__()
        keys = jr.split(key, 4)
        self.downsample_factor = downsample_factor
        self.causal = causal

        padding = ((kernel_size-1, 0),) if causal else "SAME"
        # a generalization of this would have multiple conv layers here
        self.tokenizer = eqx_modules.Conv1DLayer(
            input_dim,
            d_model,
            kernel_size = kernel_size,
            padding = padding, 
            gated = gated, 
            key = keys[0]
        )
        self.pooler = nn.AvgPool1d( 
            downsample_factor,
            downsample_factor, 
            use_ceil = True
        )
            
        self.transformer_stack = eqx_transformer.TransformerStack(
            num_layers = num_layers, 
            d_model = d_model, 
            n_heads = n_heads, 
            d_ff = d_ff,
            causal = causal, 
            key = keys[1]
        )

        self.up = nn.ConvTranspose1d(d_model, d_model, kernel_size = downsample_factor, stride = downsample_factor, padding = "same", key = keys[2]) 

        # a generalization of this would have multiple conv layers here
        padding = ((final_kernel_size-1, 0),) if causal else "SAME"
        self.final = nn.Conv1d(d_model, output_dim, final_kernel_size, padding = padding, key = keys[3])

    def __call__(self, x):
        _, L = x.shape
        char_x = self.tokenizer(x)
        x = self.pooler(char_x)
        x = x.transpose() # D x L -> L x D
        x = self.transformer_stack(x)
        x = x.transpose() # L x D -> D x L
        x = self.up(x)
        if self.causal: 
            x = jnp.pad(x, ((0,0),(self.downsample_factor-1,0)))
        x = x[:,:L] + char_x # is this guaranteed to come out length L or greater? 
        x = jax.nn.relu(x) 
        x = self.final(x) 
        return x
