import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import einops
import functools

import pallas_scan.pallas as plscan
from jax.experimental import shard_map
import einops
from jax.sharding import Mesh, PartitionSpec as P

import jax.experimental.mesh_utils as mesh_utils
import jax.random as jr
import jax.sharding as jshard

@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def sqrt_bound_derivative(
    x: jax.Array,
    max_gradient: float | jax.Array,
) -> jax.Array:
  """Computes a square root with a gradient clipped at `max_gradient`."""
  del max_gradient  # unused
  return jnp.sqrt(x)

def stable_sqrt_fwd(
    x: jax.Array,
    _: float | jax.Array
) -> tuple[jax.Array, tuple[jax.Array]]:  # pylint: disable=g-one-element-tuple
  return jnp.sqrt(x), (x,)

def stable_sqrt_bwd(
    max_gradient: float | jax.Array,
    res: tuple[jax.Array],  # pylint: disable=g-one-element-tuple
    g: jax.Array,
) -> tuple[jax.Array]:  # pylint: disable=g-one-element-tuple
  (x,) = res
  x_pre = jnp.maximum(x, 1 / (4 * max_gradient**2))
  return jax.vjp(jnp.sqrt, x_pre)[1](g)

sqrt_bound_derivative.defvjp(stable_sqrt_fwd, stable_sqrt_bwd)

class BlockDiagonalLinear(eqx.Module):

    num_heads: int
    w: jax.Array
    b: jax.Array

    def __init__(self, width: int, num_heads: int, w_init_variance_scale: float = 1.0, key=None):

        assert width % num_heads == 0
        block_width = width // num_heads

        self.num_heads = num_heads

        key_w, key_b = jax.random.split(key)
        self.w = jax.random.normal(key_w, (num_heads, block_width, block_width)) * jnp.sqrt(w_init_variance_scale / block_width)
        self.b = jnp.zeros((num_heads, block_width))

    def __call__(self, x: jax.Array) -> jax.Array:
        
        # Split x into blocks
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_heads)
        # Linear transformation over each block
        y = jnp.einsum("... h i, h i j -> ... h j", x, self.w) + self.b
        # Flatten the output
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_heads)


class RGLRU(eqx.Module):
    width: int
    num_heads: int
    a_param: jax.Array
    input_gate: eqx.Module
    a_gate: eqx.Module
    power: float
    shard_map_kwargs: dict | None

    def __init__(self, width: int, num_heads: int, w_init_variance_scale: float = 1.0, min_rad=0.9, max_rad=0.999, power=8.0, key=None, shard_map_kwargs = None):

        self.width = width
        self.num_heads = num_heads
        self.power = power
        self.shard_map_kwargs = shard_map_kwargs

        key_a, key_input, key_a_gate = jax.random.split(key, 3)

        unif = jax.random.uniform(key_a, shape=(width,))
        a_real = 0.5 * jnp.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + 1e-8)
        self.a_param = jnp.log(jnp.exp(-a_real) - 1.0) # inverse of softplus
        
        self.input_gate = BlockDiagonalLinear(width, num_heads, w_init_variance_scale, key=key_input)
        self.a_gate = BlockDiagonalLinear(width, num_heads, w_init_variance_scale, key=key_a_gate)

    def __call__(self, x: jax.Array, h0 = None) -> jax.Array:
        bs, l, _ = x.shape # so x does have batch dim aleady

        gate_x = jax.nn.sigmoid(self.input_gate(x))
        gate_a = jax.nn.sigmoid(self.a_gate(x))

        log_a = -self.power * gate_a * jax.nn.softplus(self.a_param)
        a = jnp.exp(log_a)
        a_squared = jnp.exp(2. * log_a)

        gated_x = x * gate_x
        
        multiplier = sqrt_bound_derivative(1 - a_squared, 1000.)
        normalized_x = gated_x * multiplier

        if h0 is None: 
            h0 = jnp.zeros_like(normalized_x[:,0,:]) 
        
        if self.shard_map_kwargs is None: 
            f = plscan.lru_pallas_scan
        else: 
            f = shard_map.shard_map(plscan.lru_pallas_scan, **self.shard_map_kwargs)

        y, last_h = f(
            normalized_x,
            a,
            h0
        )
    
        return y

class RecurrentBlock(eqx.Module):
    width: int
    num_heads: int
    lru_width: int
    conv1d_size: int = 4

    linear_y: eqx.Module
    linear_x: eqx.Module
    linear_out: eqx.Module
    conv_1d: eqx.Module
    lru: eqx.Module

    def __init__(self, width: int, num_heads: int, lru_width: int = None, conv1d_size: int = 4, key=None, shard_map_kwargs = None, **kwargs):

        
        lru_width = lru_width or width
        self.width = width
        self.num_heads = num_heads
        self.lru_width = lru_width
        self.conv1d_size = conv1d_size

        key_y, key_x, key_out, key_conv, key_lru = jax.random.split(key, 5)
        
        self.linear_y = nn.Linear(width, lru_width, key=key_y)
        self.linear_x = nn.Linear(width, lru_width, key=key_x)
        self.linear_out = nn.Linear(lru_width, width, key=key_out)
        self.conv_1d = nn.Sequential([
            nn.Lambda(jnp.transpose),
            nn.Conv1d(
                in_channels=lru_width,
                out_channels=lru_width,
                kernel_size=conv1d_size,
                padding=((conv1d_size-1, 0),),
                key=key_conv,
            ),
            nn.Lambda(jnp.transpose)])
        
        self.lru = RGLRU(lru_width, num_heads, key=key_lru, shard_map_kwargs = shard_map_kwargs, **kwargs)

    def __call__(self, x: jax.Array, h0 = None) -> jax.Array:
        # y branch
        y = jax.vmap(jax.vmap(self.linear_y))(x)
        y = jax.nn.gelu(y)

        # x branch
        x = jax.vmap(jax.vmap(self.linear_x))(x)

        x = jax.vmap(self.conv_1d)(x)
        
        x = self.lru(x, h0)

        # Join branches
        x = x * y
        return jax.vmap(jax.vmap(self.linear_out))(x)

class MLPBlock(eqx.Module):

    ffw_up_1: eqx.Module
    ffw_up_2: eqx.Module
    ffw_down: eqx.Module

    def __init__(self, width: int, expanded_width = None, key=None):

        expanded_width = expanded_width or width
        
        key_up_1, key_up_2, key_down = jax.random.split(key, 3)

        self.ffw_up_1 = nn.Linear(width, expanded_width, key=key_up_1)
        self.ffw_up_2 = nn.Linear(width, expanded_width, key=key_up_2)

        self.ffw_down = nn.Linear(expanded_width, width, key=key_down)

    def __call__(self, x: jax.Array) -> jax.Array:
        out_1 = jax.vmap(jax.vmap(self.ffw_up_1))(x)
        out_2 = jax.vmap(jax.vmap(self.ffw_up_2))(x)

        gate_value = jax.nn.gelu(out_1)

        return jax.vmap(jax.vmap(self.ffw_down))(gate_value * out_2)

class ResidualBlock(eqx.Module):

    temporal_pre_norm: eqx.Module
    recurrent_block: eqx.Module
    flip_block: eqx.Module | None
    channel_pre_norm: eqx.Module
    mlp: eqx.Module

    def __init__(
        self, 
        width: int, 
        num_heads: int, 
        bidir: bool = False, 
        mlp_width: int | None = None, 
        lru_width: int | None = None,
        conv1d_size: int = 4, 
        shard_map_kwargs = None, 
        key=None,
        **kwargs
    ):
        
        key_temporal, key_mlp = jax.random.split(key)

        self.temporal_pre_norm = nn.RMSNorm(width)

        self.recurrent_block = RecurrentBlock(
            width=width,
            num_heads=num_heads,
            lru_width=lru_width,
            conv1d_size=conv1d_size,
            shard_map_kwargs=shard_map_kwargs,
            key=key_temporal,
            **kwargs
        )

        self.flip_block = RecurrentBlock(
            width=width,
            num_heads=num_heads,
            lru_width=lru_width,
            conv1d_size=conv1d_size,
            shard_map_kwargs=shard_map_kwargs,
            key=key_temporal
        ) if bidir else None
        
        self.channel_pre_norm = nn.RMSNorm(width)

        self.mlp = MLPBlock(
            width=width,
            expanded_width=mlp_width,
            key=key_mlp
        )

    def __call__(self, x: jax.Array, h0 = None):
        raw_x = x

        inputs_normalized = jax.vmap(jax.vmap(self.temporal_pre_norm))(raw_x)
        x = self.recurrent_block(inputs_normalized, h0)

        if self.flip_block is not None: 
            input_flipped = jnp.flip(inputs_normalized, axis=1)
            x += jnp.flip(self.flip_block(input_flipped, h0), axis=1)

        residual = x + raw_x
        x = jax.vmap(jax.vmap(self.channel_pre_norm))(residual)

        return self.mlp(x) + residual



class RecurrentGemmaModel(eqx.Module):
    stack: list
    input_conv: nn.Sequential
    output_conv: nn.Sequential
    normalization: nn.RMSNorm
    context_embeddings: list
    
    def __init__(
        self,
        in_channels, 
        out_channels,
        kernel_size,
        num_layers,
        d_model, 
        key,
        num_heads = 4, 
        context_dims = [],
        bidir = False, # not implemented yet
        shard_map_kwargs = None,
        **kwargs
    ):

        key, in_key, out_key, *subkeys = jax.random.split(key, num_layers + 3)
        #mixer_cls = BidirMambaBlock if bidir else MambaBlock
        mixer_cls = ResidualBlock
        self.stack = [
            ResidualBlock(
                d_model,
                num_heads, 
                bidir=bidir,
                key=k,
                shard_map_kwargs = shard_map_kwargs,
                **kwargs
            )
            for k in subkeys
        ]
        self.normalization = nn.RMSNorm(d_model)
        padding = "SAME" if bidir else ((kernel_size-1, 0),) # causal conv
        self.input_conv = nn.Sequential([
            nn.Conv1d( in_channels, d_model, kernel_size, padding = padding, key=in_key ),
            nn.Lambda(jnp.transpose)
        ])
        # could try to tie in and out (and use convtranspose for out?)
        self.output_conv = nn.Sequential([
            nn.Lambda(jnp.transpose),
            nn.Conv1d(d_model, out_channels, kernel_size, padding=padding, use_bias = False, key = out_key) 
        ])

        embedding_keys = jax.random.split(key, len(context_dims))
        self.context_embeddings = [
            nn.Embedding(
                num_embeddings = context_dim, 
                embedding_size = d_model, 
                key = embedding_keys[i])
            for i,context_dim in enumerate(context_dims)
        ]

    def __call__(
        self,
        x,
        context = None,
        key = None,
    ):  
        if context is not None: 
            context_embeds = [ jax.vmap(embedding)(context[i]) for i,embedding in enumerate(self.context_embeddings) ]
            h0 = jnp.sum(jnp.stack(context_embeds), axis=0)
        else:
            h0=None
        x = jax.vmap(self.input_conv)(x)
        for layer in self.stack: 
            x = layer(x, h0=h0) 
        x = jax.vmap(jax.vmap(self.normalization))(x)
        return jax.vmap(self.output_conv)(x)


if __name__=="__main__": 

    key = jax.random.PRNGKey(0)
    width = 128
    num_heads = 4
    in_channels = 4
    out_channels = 4
    batch_size = 16
    seq_len = 1024

    context_dims = [12,20]

    x = jax.random.normal(key, (batch_size, in_channels, seq_len))
    context = [ 
        jax.random.randint(key, shape=(batch_size,), minval=0, maxval=context_dim)
        for context_dim in context_dims ]

    num_devices = len(jax.devices())
    devices = mesh_utils.create_device_mesh((num_devices,1))
    sharding = jshard.PositionalSharding(devices)
    rep_sharding = sharding.replicate()
    mesh = Mesh(devices, axis_names=('i', 'j'))

    shard_map_kwargs = {
        "mesh" : mesh,
        "in_specs" : (P("i",None,None),P("i",None,None),P("i",None)), 
        "out_specs" : (P("i",None,None),P("i",None)),
        "check_rep" : False
    }

    model_key = jax.random.PRNGKey(1)
    model = RecurrentGemmaModel(
        in_channels = 4, out_channels = 4, kernel_size = 7, d_model = width, num_layers = 3, context_dims = context_dims, bidir = True,
        key=model_key, shard_map_kwargs = shard_map_kwargs)

    x_sharded = eqx.filter_shard(x, rep_sharding)
    model_sharded = eqx.filter_shard(model, rep_sharding)
    output = model_sharded(x_sharded, context = context)
    print(output.shape)
    print(output.sum())


