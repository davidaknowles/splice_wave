import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import jax.random as jr
import jaxtyping as jt
from typing import Optional
from functools import partial

class GatedConv1d(eqx.Module):
    conv_f: nn.Conv1d
    conv_g: nn.Conv1d

    def __init__(self, *args, key, **kwargs):
        key1, key2 = jax.random.split(key)
        self.conv_f = nn.Conv1d(*args, key=key1, **kwargs)
        self.conv_g = nn.Conv1d(*args, key=key2, **kwargs)

    def __call__(self, x):
        f = self.conv_f(x)
        g = self.conv_g(x)
        return f * jax.nn.sigmoid(g)

class Conv1DLayer(eqx.Module):
    conv: nn.Conv1d
    activation: callable

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, activation=jax.nn.relu, padding = "SAME", key=None, gated = False):
        conv_cls = GatedConv1d if gated else nn.Conv1d
        self.conv = conv_cls(in_channels, out_channels, kernel_size, stride, padding=padding, key=key)
        self.activation = activation

    def __call__(self, x):
        return self.activation(self.conv(x))

class ResConv1DLayer(eqx.Module):
    conv: nn.Conv1d
    activation: callable

    def __init__(self, channels, kernel_size, stride, activation=jax.nn.relu, padding = "SAME", key=None, gated = False):
        conv_cls = GatedConv1d if gated else nn.Conv1d
        self.conv = conv_cls(channels, channels, kernel_size, stride, padding=padding, key=key)
        self.activation = activation

    def __call__(self, x):
        return self.activation(self.conv(x)) + x

class FullyConvSeq2Seq(eqx.Module):
    layers: list

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, kernel_size=3, stride=1, key=None, gated = False, causal = False):

        keys = jr.split(key, num=num_layers + 1)
        padding = ((kernel_size-1, 0),) if causal else "SAME"
        self.layers = []
        self.layers.append(Conv1DLayer(in_channels, hidden_channels, kernel_size, stride, key=keys[0], padding = padding, gated = gated))
        for i in range(1, num_layers):
            self.layers.append(ResConv1DLayer(hidden_channels, kernel_size, stride, key=keys[i], padding = padding, gated = gated))
        self.layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size, stride, padding = padding, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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
        causal = False, 
        **kwargs,
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

    def __call__(
        self,
        x: jt.Int[jt.Array, "max_seq_len input_dim"],
        mask: str | None = "causal",
        **kwargs,
    ):
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

    def __init__(self, num_layers, n_heads, d_model, d_ff, *, key, causal = False):
        keys = jax.random.split(key, num_layers)
        self.blocks = [TransformerBlock(n_heads, d_model, d_ff, key=k, causal = causal) for k in keys]

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Transformer(eqx.Module): 
    transformer_stack: TransformerStack
    input_conv: Conv1DLayer
    output_conv: nn.Conv1d
    
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, n_heads, d_model, d_ff, *, key, causal = False, input_gated = False): 
        keys = jax.random.split(key, 3)
        self.transformer_stack = TransformerStack(num_layers, n_heads, d_model, d_ff, causal = causal, key = keys[0])

        padding = ((kernel_size-1, 0),) if causal else "SAME"
        self.input_conv = Conv1DLayer(in_channels, d_model, kernel_size, padding=padding, gated = input_gated, key = keys[1])
        self.output_conv = nn.Conv1d(d_model, out_channels, kernel_size, padding=padding, key = keys[2])

    def __call__(self, x):
        x = self.input_conv(x)
        x = jnp.transpose(x) 
        x = self.transformer_stack(x)
        x = jnp.transpose(x)
        return self.output_conv(x)

if __name__=="__main__": 
    # Define parameters
    num_layers = 12    # Number of transformer blocks (BERT-base has 12)
    d_model = 768      # Hidden size (BERT-base has 768)
    n_heads = 12       # Number of attention heads (BERT-base has 12)
    d_ff = 3072        # Size of the feedforward layer (BERT-base has 3072)
    seq_length = 128   # Sequence length
    batch_size = 32    # Batch size
    
    # Random key for initialization
    key = jax.random.PRNGKey(0)

    if False: 
        # Initialize the transformer stack
        transformer = TransformerStack(num_layers, n_heads, d_model, d_ff, key=key)
        
        # Dummy input data (batch_size, seq_length, d_model)
        x = jax.random.normal(key, (batch_size, seq_length, d_model))
        
        # Run the transformer stack
        output = jax.vmap(partial(transformer, key=key))(x)
        print(output.shape)

    causal = True
    
    # full model
    key = jax.random.PRNGKey(0)
    in_channels = 4
    out_channels = 4
    model = Transformer(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 7, 
        num_layers = num_layers, 
        n_heads = n_heads, 
        d_model = d_model, 
        d_ff = d_ff, 
        causal = causal,
        key = key
    )

    x = jax.random.normal(key, (batch_size, in_channels, seq_length))
    output = jax.vmap(model)(x)
    print(output.shape)