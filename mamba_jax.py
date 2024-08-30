from typing import Optional
import math
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


def selective_scan(
    x: Float[Array, "seq_length d_inner"],
    delta: Float[Array, "seq_length d_inner"],
    A: Float[Array, "d_inner d_state"],
    B: Float[Array, "seq_length d_state"],
    C: Float[Array, "seq_length d_state"],
    D: Float[Array, " d_inner"],
) -> Float[Array, "seq_length d_inner"]:
    L, d_inner = x.shape
    _, d_state = A.shape
    delta_A = jnp.exp(jnp.einsum("l d,d n -> l d n", delta, A))
    delta_B_u = jnp.einsum("l d,l n,l d -> l d n", delta, B, x)

    x_res = jnp.zeros(shape=(d_inner, d_state))

    def step(x, i):
        x = delta_A[i] * x + delta_B_u[i]

        y = jnp.einsum("d n,n -> d", x, C[i, :])
        return x, y

    _, ys = jax.lax.scan(step, x_res, jnp.arange(L))

    ys = ys + x * D
    return ys


class SelectiveStateSpaceModel(eqx.Module):
    input_proj: nn.Linear
    delta_proj: nn.Linear
    A_log: Float[Array, "d_inner d_state"]
    D: Float[Array, " d_inner"]

    d_inner: int = eqx.field(static=True)
    dt_rank: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)

    def __init__(
        self,
        d_inner: int,
        dt_rank: int,
        d_state: int,
        use_input_proj_bias: bool = False,
        use_delta_proj_bias: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        (
            key,
            input_proj_key,
            delta_proj_key,
        ) = jax.random.split(key, 3)
        self.input_proj = nn.Linear(
            d_inner,
            dt_rank + d_state * 2,
            use_bias=use_input_proj_bias,
            key=input_proj_key,
        )

        self.delta_proj = nn.Linear(
            dt_rank, d_inner, use_bias=use_delta_proj_bias, key=delta_proj_key
        )
        A = jnp.repeat(jnp.arange(1, d_state + 1), d_inner).reshape(d_inner, d_state)
        self.A_log = jnp.log(A)
        self.D = jnp.ones(d_inner)

    def __call__(self, x: Float[Array, "seq_length d_inner"]):
        A = -jnp.exp(self.A_log)
        D = self.D

        delta_b_c = jax.vmap(self.input_proj)(x)

        split_indices = [
            self.dt_rank,
            self.dt_rank + self.d_state,
        ]
        delta, B, C = jnp.split(delta_b_c, split_indices, axis=-1)
        delta = jax.nn.softplus(jax.vmap(self.delta_proj)(delta))

        y = selective_scan(x, delta, A, B, C, D)
        return y


class MambaBlock(eqx.Module):
    in_proj: nn.Linear
    conv1d: nn.Conv1d
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
        *,
        key: PRNGKeyArray,
    ):
        dt_rank = math.ceil(d_model / 16) if (dt_rank is None) else dt_rank
        d_inner = expand * d_model
        
        (
            key,
            linear_key,
            conv1d_key,
            ssm_key,
            out_proj_key,
        ) = jax.random.split(key, 5)

        self.in_proj = nn.Linear(
            d_model,
            d_inner * 2,
            use_bias=use_in_projection_bias,
            key=linear_key,
        )

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            use_bias=use_conv_bias,
            groups=d_inner,
            padding=d_conv - 1,
            key=conv1d_key,
        )
        self.ssm = SelectiveStateSpaceModel(
            d_inner=d_inner,
            dt_rank=dt_rank,
            d_state=d_inner,
            use_delta_proj_bias=ssm_use_delta_proj_bias,
            use_input_proj_bias=ssm_use_input_proj_bias,
            key=ssm_key,
        )
        self.out_proj = nn.Linear(
            d_inner,
            d_model,
            use_bias=use_out_proj_bias,
            key=out_proj_key,
        )

    def __call__(self, x: Array):
        seq_len, d = x.shape
        x_and_res = jax.vmap(self.in_proj)(x)

        (x, res) = jnp.split(x_and_res, 2, axis=-1)
        x = jnp.transpose(x)
        x = self.conv1d(x)[:, :seq_len]
        x = jnp.transpose(x)
        x = jax.nn.silu(x)

        y = self.ssm(x)
        y = y * jax.nn.silu(res)

        output = jax.vmap(self.out_proj)(y)
        return output


class ResidualBlock(eqx.Module):
    mamba_block: MambaBlock
    rms_norm: nn.RMSNorm

    def __init__(
        self,
        d_model: int,
        key : PRNGKeyArray, 
        **kwargs
    ):
        self.mamba_block = MambaBlock(d_model, key=key, **kwargs) 
        self.rms_norm = nn.RMSNorm(d_model)

    def __call__(
        self, x: Float[Array, "seq_len d_model"], *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        return self.mamba_block(jax.vmap(self.rms_norm)(x)) + x


class Mamba(eqx.Module):
    mamba_stack: nn.Sequential
    normalization: nn.RMSNorm
    input_conv: nn.Conv1d
    output_conv: nn.Conv1d
    
    def __init__(
        self,
        in_channels, 
        out_channels,
        kernel_size,
        num_layers,
        d_model, 
        key,
        **kwargs # expand = 2, dt_rank = None, d_conv = 4
    ):
        key, in_key, out_key, *subkeys = jax.random.split(key, num_layers + 3)
        self.mamba_stack = nn.Sequential(
            [
                ResidualBlock(
                    d_model,
                    key=k,
                    **kwargs
                )
                for k in subkeys
            ],
        )
        self.normalization = nn.RMSNorm(d_model)
        padding = ((kernel_size-1, 0),) # causal conv
        self.input_conv = nn.Conv1d(
            in_channels, d_model, kernel_size, padding = padding, key=in_key
        )
        self.output_conv = nn.Conv1d(d_model, out_channels, kernel_size, padding=padding, use_bias = False, key = out_key) # could try to tie in and out (and use convtranspose for out?)

    def __call__(
        self,
        x: Int[Array, "in_channels seq_len"],  # noqa
        *,
        state: nn.State | None = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "out_channels seq_len"]:  # noqa
        
        x = self.input_conv(x)
        x = jnp.transpose(x) 
        x = self.mamba_stack(x)
        x = jax.vmap(self.normalization)(x)
        x = jnp.transpose(x)
        return self.output_conv(x)

if __name__=="__main__": 

    d_model = 32
    key = jax.random.PRNGKey(0)
    model = Mamba(
        in_channels = 4,
        out_channels = 4, 
        kernel_size = 7, 
        num_layers = 3,
        d_model = d_model,
        key = key
    )

    batch_size = 16
    seq_length = 128
    
    x = jax.random.normal(key, (4, seq_length))
    output = model(x)
    print(x.shape)

    x = jax.random.normal(key, (batch_size, 4, seq_length))
    output = jax.vmap(model)(x)
    print(x.shape)