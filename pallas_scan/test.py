import functools
from typing import Literal, overload

import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import complex_lib
import pallas

def lru_linear_scan(
    x, # B x L x ...
    a, # B x L x ...
    h0=None,
    reverse=False,
    return_a_prod=False,
    acc_float_dtype=jnp.float32,
    unroll=1,
):
    """Computes a linear scan over the second axis of the inputs.
    for t in range(x.shape[1]):
        h[:,t]=a[:,t]*h[:,t-1]+x[:,t]
    """
    acc_dtype = pallas.get_acc_dtype(x, h0, acc_float_dtype)

    def body_fn(carry, current_inputs):
        h_prev, a_prev = carry
        x_t, a_t = current_inputs
        h_t = a_t.astype(acc_dtype) * h_prev + x_t.astype(acc_dtype)
        h_out = h_t.astype(x.dtype)

        if return_a_prod:
            assert a_prev is not None
            a_t = a_t.astype(acc_dtype) * a_prev
            a_out = a_t.astype(x.dtype)
        else:
            a_t, a_out = None, None

        return (h_t, a_t), (h_out, a_out)

    h0 = complex_lib.zeros_like(x[:, 0], acc_dtype) if h0 is None else h0
    a0 = complex_lib.ones_like(h0) if return_a_prod else None

    scan_fn = jax.vmap(
        lambda init, xs: jax.lax.scan(
            body_fn,
            init=init,
            xs=xs,
            unroll=unroll,
            reverse=reverse,
        ),
        in_axes=0,
        out_axes=0,
    )
    (h_last, a_prod_last), (h, a_prod) = scan_fn((h0, a0), (x, a))

    if return_a_prod:
        return (h, h_last), (a_prod, a_prod_last)
    else:
        return (h, h_last)

if __name__=="__main__": 
    key = jax.random.PRNGKey(0)
    seq_length = 200
    d_model = 128
    batch_size = 10
    x = jax.random.normal(key, (batch_size, seq_length, d_model))
    a = jax.random.normal(key, (batch_size, seq_length, d_model))
    h, h_last = lru_linear_scan(x, a)
    print(jnp.abs(h[:,-1,:] - h_last).mean().item()) # zero! 
    print(h.shape, h_last.shape)

    h, h_last = jax.jit(pallas.lru_pallas_scan)(x, a) 
    