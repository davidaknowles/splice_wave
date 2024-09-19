from typing import Optional
import math
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
import pallas_scan.pallas as plscan
from jax.experimental import shard_map
import einops
from jax.sharding import Mesh, PartitionSpec as P

import jax.experimental.mesh_utils as mesh_utils
import jax.random as jr
import jax.sharding as jshard

def native_scan(x, a, h0 = None): 
    if h0 is None: 
        h0 = jnp.zeros_like(x[0]) 
        
    def step(h, inp): # carry, input
        x_i, a_i = inp
        h = a_i * h + x_i
        return h, h # carry, output

    h_last, h = jax.lax.scan(step, h0, (x, a))# f, init, xs
    # h : L x ED x N
    return h, h_last

def associative_scan(x, a, h0 = None):
    
    def _associative_scan_fn(s, c):
        return tuple((c[0] * s[0], c[0] * s[1] + c[1]))
    # [a, fn(a, b), fn(fn(a, b), c), ...].
    # 0: (a0,x0) -> (a
    # 1: (a0*a1, a[1]*h[0]+x[1])
    # fn(a,b) = (h1,x1) = (a1*h0+x1,x1)
    _, h = jax.lax.associative_scan(_associative_scan_fn, (a, x))
    return h, h[-1]

def selective_scan(
    x: Float[Array, "batch seq_length d_inner"],
    delta: Float[Array, "batch seq_length d_inner"],
    A: Float[Array, "d_inner d_state"], # no batch dimension
    B: Float[Array, "batch seq_length d_state"],
    C: Float[Array, "batch seq_length d_state"],
    D: Float[Array, " d_inner"], # no batch dimension
    h0 = None, # B x ED x N ? 
    shard_map_kwargs = None, 
    native = False
) -> Float[Array, "batch seq_length d_inner"]:
    # from mambapy
    # ED = expand * d_model = d_inner
    # B = batch size
    # L = seq len
    # N = d_state
    # x : (B, L, ED) 
    # Δ : (B, L, ED)
    # A : (ED, N)
    # B : (B, L, N)
    # C : (B, L, N)
    # D : (ED)

    # y : (B, L, ED)

    _, L, d_inner = x.shape
    _, d_state = A.shape

    # there's no sum here? it's just an elementwise multiply
    #delta_A = jnp.exp(jnp.einsum("b l d,d n -> b l d n", delta, A))
    delta_A = jnp.exp(delta[..., None] * A) # delta_A is B x L x ED x N
    # again not an actual sum! 
    #delta_B_x = jnp.einsum("b l d,b l n,b l d -> b l d n", delta, B, x) # B x L x ED x N
    delta_B_x = delta[..., None] * B[:, :, None, :] * x[:, :, :, None]

    if native: 
        if h0 is not None: 
            h0 = einops.rearrange(delta_A, 'b (d n) -> b d n', d=d_inner, n=d_state)
        h, _ = jax.vmap(native_scan)(delta_B_x, delta_A, h0 = h0)
        #h, _ = jax.vmap(associative_scan)(delta_B_x, delta_A) # even slower! 
    else:
        if shard_map_kwargs is None: 
            f = plscan.lru_pallas_scan
        else: 
            f = shard_map.shard_map(plscan.lru_pallas_scan, **shard_map_kwargs)
        delta_B_x = einops.rearrange(delta_B_x, 'b l d n -> b l (d n)')
        if h0 is None: 
            h0 = jnp.zeros_like(delta_B_x[:,0]) 
        h, _ = f(
            delta_B_x,
            einops.rearrange(delta_A, 'b l d n -> b l (d n)'),
            h0
        )
        h = einops.rearrange(h, 'b l (d n) -> b l d n', d=d_inner, n=d_state)
    
    ys = (h @ C[..., None]).squeeze(-1) # ys = jnp.einsum("l d n,l n -> l d", h, C) 
    ys = ys + x * D
    return ys


class SelectiveStateSpaceModel(eqx.Module):
    input_proj: nn.Linear
    dt_proj: nn.Linear
    A_log: Float[Array, "d_inner d_state"]
    D: Float[Array, " d_inner"]

    d_inner: int = eqx.field(static=True)
    dt_rank: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)

    shard_map_kwargs: dict | None

    def __init__(
        self,
        d_inner: int,
        dt_rank: int,
        d_state: int,
        use_input_proj_bias: bool = False,
        use_delta_proj_bias: bool = False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        shard_map_kwargs = None, 
        *,
        key: PRNGKeyArray,
    ):
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.shard_map_kwargs = shard_map_kwargs
        
        key, input_proj_key,  delta_proj_key = jax.random.split(key, 3)
        
        self.input_proj = nn.Linear( # this is called x_proj in og mamba
            d_inner,
            dt_rank + d_state * 2, # provides delta, B and C
            use_bias=use_input_proj_bias,
            key=input_proj_key,
        )

        dt_proj = nn.Linear( # = dt_proj in og_mamba
            dt_rank, d_inner, use_bias=use_delta_proj_bias, key=delta_proj_key
        )
        # og mamba has quite careful initialization of dt_proj which is ignored here
        dt_init_std = dt_rank**-0.5 * dt_scale
        
        # Initialize the dt_proj weight
        if dt_init == "constant":
            dt_proj_weight = jnp.full_like(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            dt_proj_weight = jax.random.uniform(delta_proj_key, dt_proj.weight.shape, minval=-dt_init_std, maxval=dt_init_std)

        dt_proj = eqx.tree_at(lambda dt: dt.weight, dt_proj, dt_proj_weight)
        
        # Initialize dt bias
        random_vals = jax.random.uniform(delta_proj_key, (d_inner,))
        dt = jnp.exp(
            random_vals * (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)
        ).clip(min=dt_init_floor)
        
        # Inverse of softplus
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        
        # Update the dt_proj bias
        dt_proj = eqx.tree_at(lambda dt: dt.bias, dt_proj, inv_dt)
        self.dt_proj = dt_proj
        
        A = jnp.repeat(jnp.arange(1, d_state + 1), d_inner).reshape(d_inner, d_state)
        self.A_log = jnp.log(A)
        self.D = jnp.ones(d_inner) # og mamba sets D._no_weight_decay = True

    def __call__(self, x: Float[Array, "batch seq_length d_inner"], h0 = None):
        A = -jnp.exp(self.A_log)
        D = self.D

        delta_b_c = jax.vmap(jax.vmap(self.input_proj))(x) 

        split_indices = [
            self.dt_rank,
            self.dt_rank + self.d_state,
        ]

        # outputs will be seq_length x dt_rank, seq_length x d_state
        delta, B, C = jnp.split(delta_b_c, split_indices, axis=-1) # layerNorm these? 
        delta = jax.nn.softplus(jax.vmap(jax.vmap(self.dt_proj))(delta))

        #y = jax.vmap(partial(selective_scan, A=A, D=D))(x = x, delta = delta, B = B, C = C)
        y = selective_scan(x, delta, A, B, C, D, h0 = h0, shard_map_kwargs = self.shard_map_kwargs)
        return y

class Mamba(eqx.Module): # renamed for consistency with o.g. implementation
    in_proj: nn.Linear
    conv1d: nn.Sequential
    ssm: SelectiveStateSpaceModel
    out_proj: nn.Linear

    def __init__(
        self,
        d_model: int,
        dt_rank: int | None = None,
        d_conv: int = 4,
        expand: int = 2, 
        use_in_projection_bias: bool = True,
        use_conv_bias: bool = True,
        use_out_proj_bias: bool = True,
        ssm_use_delta_proj_bias: bool = False,
        ssm_use_input_proj_bias: bool = False,
        shard_map_kwargs = None,
        *,
        key: PRNGKeyArray,
    ):
        dt_rank = math.ceil(d_model / 16) if (dt_rank is None) else dt_rank
        d_inner = expand * d_model
        
        key, linear_key, conv1d_key, ssm_key, out_proj_key = jax.random.split(key, 5)

        self.in_proj = nn.Linear( # all same as og mamba
            d_model,
            d_inner * 2,
            use_bias=use_in_projection_bias,
            key=linear_key,
        )

        self.conv1d = nn.Sequential([
            nn.Lambda(jnp.transpose),
            nn.Conv1d( # all same as og mamba
                in_channels=d_inner,
                out_channels=d_inner,
                kernel_size=d_conv,
                use_bias=use_conv_bias,
                groups=d_inner,
                padding=d_conv - 1,
                key=conv1d_key,
            ),
            nn.Lambda(jnp.transpose)
        ])
        self.ssm = SelectiveStateSpaceModel(
            d_inner=d_inner,
            dt_rank=dt_rank,
            d_state=d_inner,
            use_delta_proj_bias=ssm_use_delta_proj_bias,
            use_input_proj_bias=ssm_use_input_proj_bias,
            shard_map_kwargs = shard_map_kwargs, 
            key=ssm_key,
        )
        self.out_proj = nn.Linear(  # all same as og mamba
            d_inner,
            d_model,
            use_bias=use_out_proj_bias,
            key=out_proj_key,
        )

    def __call__(self, x: Array, h0 = None):
        B, seq_len, d = x.shape
        x_and_res = jax.vmap(jax.vmap(self.in_proj))(x)

        (x, res) = jnp.split(x_and_res, 2, axis=-1)
        x = jax.vmap(self.conv1d)(x)[:, :seq_len, :]

        x = jax.nn.silu(x)

        y = self.ssm(x, h0 = h0)
        y = y * jax.nn.silu(res)

        output = jax.vmap(jax.vmap(self.out_proj))(y)
        return output


class MambaBlock(eqx.Module):
    mamba: Mamba
    norm: nn.RMSNorm
    norm_last: bool

    def __init__(
        self,
        d_model: int,
        norm_last = False, 
        layer_norm = False, 
        key : PRNGKeyArray = None, 
        shard_map_kwargs = None,
        **kwargs
    ):
        self.norm_last = norm_last
        self.mamba = Mamba(d_model, key=key, shard_map_kwargs = shard_map_kwargs, **kwargs) 
        self.norm = (nn.LayerNorm if layer_norm else nn.RMSNorm)(d_model)

    def __call__(
        self, x: Float[Array, "batch seq_len d_model"], h0 = None, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        if self.norm_last: 
            h = self.mamba(x, h0=h0) + x
            return jax.vmap(jax.vmap(self.norm))(h)
        else: 
            h = jax.vmap(jax.vmap(self.norm))(x)
            return self.mamba(h, h0=h0) + x

class BidirMambaBlock(eqx.Module):
    mamba: Mamba
    mamba_flip: Mamba
    norm: nn.RMSNorm
    norm_last: bool

    def __init__(
        self,
        d_model: int,
        norm_last = False, 
        layer_norm = False, 
        key : PRNGKeyArray = None, 
        shard_map_kwargs = None,
        **kwargs
    ):
        self.norm_last = norm_last
        self.mamba = Mamba(d_model, key=key, shard_map_kwargs = shard_map_kwargs, **kwargs) 
        self.mamba_flip = Mamba(d_model, key=key, shard_map_kwargs = shard_map_kwargs, **kwargs) 
        self.norm = (nn.LayerNorm if layer_norm else nn.RMSNorm)(d_model)

    def __call__(
        self, x: Float[Array, "batch seq_len d_model"], h0 = None, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        if self.norm_last: 
            x_flipped = jnp.flip(x, axis=1)
            h = self.mamba(x, h0=h0) + self.mamba_flip(x_flipped, h0=h0) + x
            return jax.vmap(jax.vmap(self.norm))(h)
        else: 
            h = jax.vmap(jax.vmap(self.norm))(x)
            h_flipped = jnp.flip(h, axis=1)
            return self.mamba(h, h0=h0) + self.mamba_flip(h_flipped, h0=h0) + x

class MambaModel(eqx.Module):
    mamba_stack: nn.Sequential
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
        context_dims = [],
        bidir = False, 
        norm_last = False,
        layer_norm = False, 
        shard_map_kwargs = None, 
        **kwargs # expand = 2, dt_rank = None, d_conv = 4
    ):
        key, in_key, out_key, *subkeys = jax.random.split(key, num_layers + 3)
        mixer_cls = BidirMambaBlock if bidir else MambaBlock
        #self.mamba_stack = nn.Sequential(
        self.mamba_stack = [
                mixer_cls(
                    d_model,
                    norm_last = norm_last, 
                    layer_norm = layer_norm,
                    key=k,
                    shard_map_kwargs = shard_map_kwargs, 
                    **kwargs
                )
                for k in subkeys
            ]
        # )
        self.normalization = nn.RMSNorm(d_model)
        padding = ((kernel_size-1, 0),) # causal conv
        self.input_conv = nn.Sequential([
            nn.Conv1d(
                in_channels, d_model, kernel_size, padding = padding, key=in_key
                ),
            nn.Lambda(jnp.transpose)
        ])
        self.output_conv = nn.Sequential([
            nn.Lambda(jnp.transpose),
            nn.Conv1d(d_model, out_channels, kernel_size, padding=padding, use_bias = False, key = out_key) # could try to tie in and out (and use convtranspose for out?)
        ])

        embedding_keys = jax.random.split(key, len(context_dims))
        self.context_embeddings = [
            nn.Embedding(
                num_embeddings = context_dim, 
                embedding_size = d_model**2 * 4, # assume expand==2
                key = embedding_keys[i])
            for i,context_dim in enumerate(context_dims)
        ]

    def __call__(
        self,
        x: Int[Array, "batch in_channels seq_len"],  # noqa
        context = None,
        *,
        state: nn.State | None = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "batch out_channels seq_len"]:  # noqa
        
        if context is not None: 
            context_embeds = [ jax.vmap(embedding)(context[i]) for i,embedding in enumerate(self.context_embeddings) ]
            h0 = jnp.sum(jnp.stack(context_embeds), axis=0)
        else:
            h0=None
        x = jax.vmap(self.input_conv)(x)
        #x = self.mamba_stack(x)
        for layer in self.mamba_stack: 
            x = layer(x, h0=h0) 
        x = jax.vmap(jax.vmap(self.normalization))(x)
        return jax.vmap(self.output_conv)(x)

if __name__=="__main__": 

    d_model = 32
    key = jax.random.PRNGKey(0)

    batch_size = 16
    seq_length = 128

    context_dims = [12,20]

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
    model = MambaModel(
        in_channels = 4,
        out_channels = 4, 
        kernel_size = 7, 
        num_layers = 3,
        d_model = d_model,
        context_dims = context_dims,
        shard_map_kwargs = shard_map_kwargs,
        key = key
    )

    x = jax.random.normal(key, (batch_size, 4, seq_length))
    context = [ 
        jax.random.randint(key, shape=(batch_size,), minval=0, maxval=context_dim)
        for context_dim in context_dims ]
    
    x_sharded = eqx.filter_shard(x, rep_sharding)
    model_sharded = eqx.filter_shard(model, rep_sharding)
    output = model_sharded(x_sharded, context = context)
    print(output.shape)

    if False: 
        key = jax.random.PRNGKey(0)
        seq_length = 200
        d_model = 128
        batch_size = 12
        x = jax.random.normal(key, (batch_size, seq_length, d_model))
        a = jax.random.normal(key, (batch_size, seq_length, d_model))
    
        x_sharded, a_sharded = eqx.filter_shard((x,a), rep_sharding)
    
        f = shard_map.shard_map(
            plscan.lru_pallas_scan, 
            mesh = mesh,
            in_specs = (P("i",None,None),P("i",None,None)), 
            out_specs = (P("i",None,None),P("i",None)),
            check_rep = False
        )
        h, h_last = f(x_sharded, a_sharded) 
        print(h.shape)
    
        x = jax.random.normal(key, (seq_length, d_model))
        a = jax.random.normal(key, (seq_length, d_model))
        h, h_last = native_scan(x,a)
        h_, h_last = associative_scan(x,a)
