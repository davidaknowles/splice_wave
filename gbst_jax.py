import math
from math import gcd
import functools
import equinox as eqx
from equinox import nn

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jt
import einops
from functools import partial
# helpers

def exists(val):
    return val is not None

def lcm(*numbers): 
    """least common multiple"""
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def next_divisible_length(seqlen, multiple):
    return math.ceil(seqlen / multiple) * multiple

def pad_to_multiple(tensor, multiple, *, seq_len, dim=-1, value=0.):
    length = next_divisible_length(seq_len, multiple)
    if length == seq_len:
        return tensor
    remainder = length - seq_len
    pad_widths = [(0, 0)] * tensor.ndim
    pad_widths[dim] = (0, remainder)
    return jnp.pad(tensor, pad_widths, constant_values=value)
    
class GBST(eqx.Module):
    pos_conv: nn.Conv1d
    score_fn: nn.Linear
    blocks: tuple
    block_pad_multiple: int
    downsample_factor: int

    def __init__(
        self,
        input_dim,  # vocab size
        d_model,  # model dimension
        max_block_size=None,
        blocks=None,
        downsample_factor=4,
        *,
        key
    ):
        """Deviating from the paper, you can also specify block size(s) with different offsets. This is to cover a potential use-case for genomics pre-training, where the tokenizer should be able to learn the correct frame. Simply omit the max_block_size, and pass in blocks as a list of tuples of tuples, each tuple with the format (block size, offset). Offsets must be less than the block size.

        With len(blocks)==1 I believe this is at least roughly equivalent to Conv -> MeanPooling. 
        A causal could probably be made using a bank of Conv(stride=downsample_factor)s with different kernel_sizes 
        """
        assert exists(max_block_size) ^ exists(blocks), 'either max_block_size or blocks are given on initialization'

        if exists(blocks):
            assert isinstance(blocks, tuple), 'blocks must be a tuple of block sizes'
            self.blocks = tuple(map(lambda el: el if isinstance(el, tuple) else (el, 0), blocks))
            assert all([(offset < block_size) for block_size, offset in self.blocks]), 'offset must be always smaller than the block size'
            max_block_size = max(list(map(lambda t: t[0], self.blocks)))
        else:
            self.blocks = tuple(map(lambda el: (el, 0), range(1, max_block_size + 1)))

        conv_key, linear_key = jt.split(key)
        
        self.pos_conv = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=max_block_size, padding="same", key=conv_key)

        self.score_fn = nn.Linear(d_model, 1, key=linear_key)

        assert downsample_factor <= max_block_size, 'final downsample factor should be less than the maximum block size'

        self.block_pad_multiple = lcm(*[block_size for block_size, _ in self.blocks])
        self.downsample_factor = downsample_factor

    def __call__(self, x):
        """Process a single example. x is input_dim x L"""

        _, L = x.shape
        
        # Generate positions for the tokens
        x = self.pos_conv(x)  # x is now d_model x L

        # Pad sequence to length visible by all block sizes from 0 to max block size
        x = pad_to_multiple(x, self.block_pad_multiple, seq_len=L, dim=-1)

        # Compute representations for all blocks by mean pooling
        block_reprs = []

        for block_size, offset in self.blocks:
            # Pad for offsets, if needed
            if offset > 0:
                left_offset, right_offset = (block_size - offset), offset
                x_padded = jnp.pad(x, ((0, 0), (left_offset, right_offset)))
            else:
                x_padded = x

            # Group input sequence into blocks, mean over block and implicitly transpose
            # here n indexes the blocks
            block_repr = einops.reduce(x_padded, 'd (n m) -> n d', 'mean', m=block_size)

            # Repeat block representation to get back to original sequence length
            block_repr = einops.repeat(block_repr, 'n d -> (n m) d', m=block_size)

            if offset > 0:
                block_repr = block_repr[left_offset:-right_offset]

            block_reprs.append(block_repr)
        
        # Stack all the block representations
        block_reprs = jnp.stack(block_reprs, axis=1) # output is L x len(blocks) x d_model

        # Calculate scores and softmax across the block size dimension
        # the two vmaps are to match pytorch.nn.Linear behavior, i.e. broadcasting over first two dims
        scores = jax.vmap(jax.vmap(self.score_fn))(block_reprs) # output is L x len(blocks) x 1
        # thus the (soft) choice of block size is the same for each d
        scores = jax.nn.softmax(scores, axis=1) # normalize over blocks dim

        # Multiply the block representations by the position-wise scores
        # sum over the block dimension. This is done per (position, model dim) tuple
        x = jnp.einsum('l m d, l m 1 -> l d', block_reprs, scores)

        # Get the next int >=n that is divisible by LCM (and therefore all block sizes)
        ndl = next_divisible_length(L, self.downsample_factor) # hopefully usually just L by design

        # Truncate to length divisible by downsample factor
        char_x = x[:ndl, :]

        # Final mean pooling downsample, F_D in the paper
        x = einops.reduce(char_x, '(n m) d -> n d', 'mean', m=self.downsample_factor)

        return char_x[:L, :], x

if __name__ == "__main__":

    # Parameters
    key = jt.PRNGKey(0)
    input_dim = 64  # vocab size
    d_model = 128  # model dimension
    max_block_size = 5
    downsample_factor = 4
    
    # Initialize GBST layer
    gbst = GBST(input_dim=input_dim, d_model=d_model, max_block_size=max_block_size, downsample_factor=downsample_factor, key=key)
    
    # Create dummy input data
    batch_size = 32
    seq_length = 128
    x = jt.normal(key, (batch_size, input_dim, seq_length))
    
    # Run the GBST layer
    char_x, x = jax.vmap(gbst)(x) # x out is B x seq_length x d_model
    print(char_x.shape, x.shape) 

    #x = jt.normal(key, (input_dim, seq_length))
    #char_x, x = gbst(x)
