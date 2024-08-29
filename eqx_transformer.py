import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import jax.random as jr
import jaxtyping as jt
from typing import Optional
from functools import partial

class TransformerBlock(eqx.Module):
    mha_attention: nn.MultiheadAttention
    rms_norm: nn.RMSNorm
    feedforward: nn.MLP
    causal: bool

    rope_embeddings: nn.RotaryPositionalEmbedding

    def __init__(
        self,
        num_heads: int,
        d_model: int, 
        d_ff: int,
        key,
        causal = False
    ):
        subkeys = jax.random.split(key, 2)
        self.rope_embeddings = nn.RotaryPositionalEmbedding(
            embedding_size=d_model // num_heads,
        )

        self.mha_attention = nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model, 
            key=subkeys[0]
        )

        self.rms_norm = nn.RMSNorm(shape=d_model)

        self.feedforward = nn.MLP(
            d_model,
            out_size=d_model,
            width_size=d_ff,
            depth=2,
            key=subkeys[1],
        )

        self.causal = causal

    def __call__(self, x):
        def process_heads(query_heads, key_heads, value_heads):
            query_heads = jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(query_heads)
            key_heads = jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(key_heads)

            return query_heads, key_heads, value_heads

        L, _ = x.shape

        if self.causal: # TODO: precompute this? 
            query_indices = jnp.arange(L)[:, None]
            kv_indices = jnp.arange(L)[None, :]
            mask = kv_indices <= query_indices
        else: 
            mask = None

        vmap_rms_norm = jax.vmap(self.rms_norm)
        
        mha = self.mha_attention(
            process_heads=process_heads,
            query=vmap_rms_norm(x),
            key_=vmap_rms_norm(x),
            value=vmap_rms_norm(x),
            mask=mask
        )
 
        x = mha + x
        normed_x = vmap_rms_norm(x)
        ff = jax.vmap(self.feedforward)(normed_x)
        x = ff + x
        return x

class TransformerStack(eqx.Module):
    blocks: list

    def __init__(
        self, 
        num_layers, 
        n_heads, 
        d_model, 
        d_ff, 
        *, 
        key, 
        causal = False
    ):
        keys = jax.random.split(key, num_layers)
        self.blocks = [TransformerBlock(n_heads, d_model, d_ff, key=k, causal = causal) for k in keys]

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x

if __name__=="__main__": 
    # Define parameters
    num_layers = 12    # Number of transformer blocks (BERT-base has 12)
    d_model = 768      # Hidden size (BERT-base has 768)
    n_heads = 12       # Number of attention heads (BERT-base has 12)
    d_ff = 3072        # Size of the feedforward layer (BERT-base has 3072)
    seq_length = 1024   # Sequence length
    batch_size = 128    # Batch size
    
    # Random key for initialization
    key = jax.random.PRNGKey(0)

    # Initialize the transformer stack
    transformer = TransformerStack(num_layers, n_heads, d_model, d_ff, key=key, causal = False)
    
    # Dummy input data (batch_size, seq_length, d_model)
    x = jax.random.normal(key, (batch_size, seq_length, d_model))
    
    # Run the transformer stack
    output = jax.vmap(transformer)(x)
    print(output.shape)
