# lucidrains implementation of Charformer, based on the paper "Charformer: Fast Character Transformers via Gradient-Based Subword Tokenization"

import math
from math import gcd
import functools
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def lcm(*numbers): 
    """least common multiple"""
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def masked_mean(tensor, mask, dim = -1):
    """This will be inefficient on TPU, but is only relevant if masking"""

    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim = dim)
    mean = tensor.sum(dim = dim) / total_el.clamp(min = 1.)
    mean.masked_fill_(total_el == 0, 0.)
    return mean

def next_divisible_length(seqlen, multiple):
    return math.ceil(seqlen / multiple) * multiple

def pad_to_multiple(tensor, multiple, *, seq_len, dim = -1, value = 0.):
    length = next_divisible_length(seq_len, multiple)
    if length == seq_len:
        return tensor
    remainder = length - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

# main class
class GBST(nn.Module):
    def __init__(
        self,
        input_dim, # vocab size
        d_model, # model dimension
        max_block_size = None,
        blocks = None,
        downsample_factor = 4,
        score_consensus_attn = True
    ):
        """Deviating from the paper, you can also specify block size(s) with different offsets. This is to cover a potential use-case for genomics pre-training, where the tokenizer should be able to learn the correct frame. Simply omit the max_block_size, and pass in blocks as a list of tuples of tuples, each tuple with the format (block size, offset). Offsets must be less than the block size"""
        super().__init__()
        assert exists(max_block_size) ^ exists(blocks), 'either max_block_size or blocks are given on initialization'

        if exists(blocks):
            assert isinstance(blocks, tuple), 'blocks must be a tuple of block sizes'
            self.blocks = tuple(map(lambda el: el if isinstance(el, tuple) else (el, 0), blocks))
            assert all([(offset < block_size) for block_size, offset in self.blocks]), 'offset must be always smaller than the block size'
            max_block_size = max(list(map(lambda t: t[0], self.blocks)))
        else:
            self.blocks = tuple(map(lambda el: (el, 0), range(1, max_block_size + 1)))

        self.pos_conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size = max_block_size, padding = "same"),
            Rearrange('b d n -> b n d')
        )

        self.score_fn = nn.Sequential( # F_R in the paper
            nn.Linear(d_model, 1),
            Rearrange('... () -> ...') # flatten
        )

        self.score_consensus_attn = score_consensus_attn

        assert downsample_factor <= max_block_size, 'final downsample factor should be less than the maximum block size'

        self.block_pad_multiple = lcm(*[block_size for block_size, _ in self.blocks])
        self.downsample_factor = downsample_factor

    def forward(self, x, mask = None, L = None):
        """x is B x D x L"""

        if L is None: 
            L = x.shape[2] # try to avoid this on TPU

        # do a conv to generate the positions for the tokens
        x = self.pos_conv(x) # still B x L x D

        # pad both sequence and mask to length visibile by all block sizes from 0 to max block size
        x = pad_to_multiple(x, self.block_pad_multiple, seq_len = L, dim = -2)

        if exists(mask):
            mask = pad_to_multiple(mask, self.block_pad_multiple, seq_len = L, dim = -1, value = False)

        # compute representations for all blocks by mean pooling
        block_masks = []
        block_reprs = []

        for block_size, offset in self.blocks:
            # clone the input sequence as well as the mask, in order to pad for offsets

            block_x = x.clone()

            if exists(mask):
                block_mask = mask.clone()

            # pad for offsets, if needed
            need_padding = offset > 0

            if need_padding:
                left_offset, right_offset = (block_size - offset), offset
                block_x = F.pad(block_x, (0, 0, left_offset, right_offset), value = 0.)

                if exists(mask):
                    block_mask = F.pad(block_mask, (left_offset, right_offset), value = False)

            # group input sequence into blocks

            blocks = rearrange(block_x, 'b (n m) d -> b n m d', m = block_size)

            # either mean pool the blocks, or do a masked mean
            if exists(mask):
                mask_blocks = rearrange(block_mask, 'b (n m) -> b n m', m = block_size)
                block_repr = masked_mean(blocks, mask_blocks, dim = -2)
            else:
                block_repr = blocks.mean(dim = -2)

            # append the block representations, as well as the pooled block masks

            block_repr = repeat(block_repr, 'b n d -> b (n m) d', m = block_size)

            if need_padding:
                block_repr = block_repr[:, left_offset:-right_offset]

            block_reprs.append(block_repr)

            if exists(mask):
                mask_blocks = torch.any(mask_blocks, dim = -1)
                mask_blocks = repeat(mask_blocks, 'b n -> b (n m)', m = block_size)

                if need_padding:
                    mask_blocks = mask_blocks[:, left_offset:-right_offset]

                block_masks.append(mask_blocks)

        # stack all the block representations

        block_reprs = torch.stack(block_reprs, dim = 2)

        # calculate scores and softmax across the block size dimension

        scores = self.score_fn(block_reprs)

        if exists(mask):
            block_masks = torch.stack(block_masks, dim = 2)
            max_neg_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(~block_masks, max_neg_value)

        scores = scores.softmax(dim = 2)

        # do the cheap consensus attention, eq (5) in paper

        if self.score_consensus_attn:
            score_sim = einsum('b i d, b j d -> b i j', scores, scores)

            if exists(mask):
                cross_mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
                max_neg_value = -torch.finfo(score_sim.dtype).max
                score_sim = score_sim.masked_fill(~cross_mask, max_neg_value)

            score_attn = score_sim.softmax(dim = -1)
            scores = einsum('b i j, b j m -> b i m', score_attn, scores)

        # multiply the block representations by the position-wise scores

        scores = rearrange(scores, 'b n m -> b n m ()')
        x = (block_reprs * scores).sum(dim = 2)

        # get the next int >=n that is divisible by LCM (and therefore all block sizes)
        m = next_divisible_length(L, self.downsample_factor) 

        # truncate to length divisible by downsample factor
        char_x = x[:, :m]

        if exists(mask):
            mask = mask[:, :m]

        # final mean pooling downsample, F_D in the paper
        x = rearrange(char_x, 'b (n m) d -> b n m d', m = self.downsample_factor)

        if exists(mask):
            mask = rearrange(mask, 'b (n m) -> b n m', m = self.downsample_factor)
            x = masked_mean(x, mask, dim = 2)
            mask = torch.any(mask, dim = -1)
        else:
            x = x.mean(dim = -2)

        return char_x[:,:L], x, mask

class GBST_OG(nn.Module): 

    def __init__(self, num_tokens, d_model, **kwargs): 
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.gbst = GBST(input_dim = d_model, d_model = d_model, **kwargs)

    def forward(self, x): 
        # get character token embeddings
        x = self.token_emb(x) # now x is B x L x D
        x = x.transpose(1,2)
        return self.gbst(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=5000):
        super().__init__()
        self.d_model = d_model

        # Create a positional encoding matrix of size (max_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply the sine to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cosine to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add a batch dimension (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape (batch_size, seq_len, d_model)
        return x + self.pe

class Charformer(nn.Module): 

    def __init__(
        self, 
        input_dim, 
        d_model, 
        output_dim,
        downsample_factor, 
        mixer_cls, 
        seq_len, 
        max_block_size = None, 
        blocks = None, 
        num_layers = 3, 
        upsample_with_conv = False, 
        final_kernel_size = 7
    ): 
        super().__init__()
        self.pe = PositionalEncoding(
            d_model, 
            seq_len = seq_len // downsample_factor
        )
        self.tokenizer = GBST(
            input_dim = input_dim,
            d_model = d_model,                    # dimension of token and intra-block positional embedding
            max_block_size = max_block_size,           # maximum block size
            blocks = blocks,
            downsample_factor = downsample_factor,        # the final downsample factor by which the sequence length will decrease by
            score_consensus_attn = False   # whether to do the cheap score consensus (aka attention) as in eq. 5 in the paper
        )
        self.mixer = nn.Sequential(*[ # better to use TransformerEncoder? 
            mixer_cls(d_model = d_model, batch_first = True)
         for _ in range(num_layers)])

        self.up = nn.Sequential(
            Rearrange('b n d -> b d n'), 
            nn.ConvTranspose1d(d_model, d_model, kernel_size = downsample_factor, stride = downsample_factor) 
            if upsample_with_conv else nn.Upsample(scale_factor=downsample_factor), 
        )

        self.final = nn.Conv1d(d_model, output_dim, final_kernel_size, bias = False, padding = "same")

    def forward(self, x, L = None): 
        char_x, x, _ = self.tokenizer(x, L = L) 
        x = self.pe(x)
        x = self.mixer(x)
        x = self.up(x) + char_x.transpose(1,2)
        x = F.relu(x) 
        x = self.final(x) 
        return x
        

if __name__ == "__main__":

    from functools import partial
    
    vocab_size = 4
    seq_len = 100

    mixer_cls = partial(
        nn.TransformerEncoderLayer,
        nhead = 4, 
        dim_feedforward = 64, 
        dropout = 0., 
        bias = True # try false
    )

    model = Charformer(
        seq_len = seq_len, 
        input_dim = vocab_size, 
        d_model = 32, 
        output_dim = vocab_size,
        downsample_factor = 4, 
        #max_block_size = 5, 
        blocks = ((1,0),(3,0),(5,0)),
        mixer_cls = mixer_cls
    )

    batch_size = 3
    tokens = torch.rand(batch_size, vocab_size, seq_len) 
    
    # both tokens and mask will be appropriately downsampled
    
    x = model(tokens, L = seq_len)
    print(x.shape)

