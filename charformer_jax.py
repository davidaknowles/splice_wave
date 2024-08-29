import jax_dataloader as jdl
import epigenome_data
from typing import cast
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import jax.experimental.mesh_utils as mesh_utils
import jax.random as jr
import jax.sharding as jshard
import optax
from functools import partial
from jax.scipy.special import logsumexp
from utils import RateTracker
import pandas as pd
from pathlib import Path
import time
import eqx_modules
import eqx_transformer
import gbst_jax
import equinox.nn as nn

MLM = True
MODEL = 'Charformer'

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
        print(x.shape)
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
        print(x.shape)
        x = x.transpose() # L x D -> D x L
        x = self.up(x)[:,:L] + char_x.transpose() # is this guaranteed to come out length L or greater? 
        x = jax.nn.relu(x) 
        x = self.final(x) 
        return x

#@eqx.filter_value_and_grad
def compute_loss(model, data):
    if MLM: 
        one_hot_T, x, mask = data
    else: 
        x = data
    output = jax.vmap(model)(x)
    out_norm = output - logsumexp(output, axis=1, keepdims=True)
    #seq_mask = mask[:, None, 1:].astype(jnp.float32) # could just sum one_hot_masked_T instead
    if MLM: 
        seq_mask = mask[:, None, :].astype(jnp.float32) 
        loss = - (seq_mask * one_hot_T * out_norm).sum() / (seq_mask.sum() + 1e-8)
    else: 
        loss = - (x[:, :, 1:] * out_norm[:, :, :-1]).sum() / (x[:, :, 1:].sum() + 1e-8) # naturally accounts for missing
    return loss

@eqx.filter_jit(donate="all-except-first")
def evaluate(model, data, rep_sharding):
    model = eqx.filter_shard(model, rep_sharding)
    data = eqx.filter_shard(data, rep_sharding)
    return compute_loss(model, data)

@eqx.filter_jit(donate="all")
def train_step(model, opt_state, data, rep_sharding = None):
    if rep_sharding is not None: 
        model, opt_state = eqx.filter_shard((model, opt_state), rep_sharding)

    # patrick says this would just act as an assertion here
    #one_hot_T, one_hot_masked_T, mask = eqx.filter_shard((one_hot_T, one_hot_masked_T, mask), replicated)

    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, data)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    if rep_sharding is not None: 
        model, opt_state = eqx.filter_shard((model, opt_state), rep_sharding)

    return loss, model, opt_state

def loop(dataloader, model, rep_sharding, opt_state = None, print_every = 10): 

    start_time = time.time()
    
    losses = []
    tracker = RateTracker()
    
    for step, batch in enumerate(dataloader):
        if MLM: 
            species, tissue, assay, one_hot, one_hot_masked, mask = batch
            one_hot_masked_T = np.swapaxes(one_hot_masked, 1, 2)
            one_hot_T = np.swapaxes(one_hot, 1, 2)
            data = (one_hot_T, one_hot_masked_T, mask)
        else: 
            species, tissue, assay, one_hot = batch
            data = np.swapaxes(one_hot, 1, 2)
    
        data = eqx.filter_shard(data, rep_sharding)

        if opt_state is None: 
            loss = evaluate(model, data, rep_sharding)
        else: 
            loss, model, opt_state = train_step(model, opt_state, data, rep_sharding)
        
        loss = loss.item()
        
        losses.append(loss)

        tracker.add(batch_size)

        elapsed = time.time() - start_time
        if elapsed > print_every: 
            start_time = time.time()
            print(f"Epoch:{epoch} step:{step}, loss:{loss} rate:{tracker.rate()}")
    
    epoch_loss = np.mean(losses)
    return model, opt_state, epoch_loss

sequence_len = 1024

num_workers = 50


if MODEL == "Conv": 
    batch_size = 1024 
    # fully convolutional network, no transformers or similar so limited receptive field
    model = eqx_modules.FullyConvSeq2Seq(
        in_channels = 4, 
        hidden_channels = 128, 
        out_channels = 4, 
        num_layers = 5, 
        kernel_size = 7, 
        gated = True,
        causal = not MLM,
        key = jr.PRNGKey(0)
    )
elif MODEL == "Transformer": 
    batch_size = 128
    # convolutional input and output layers, and transformer (with RoPE) stack inbetween _at full nucleotide resolution_ which is likely inefficient computationally and not a great modeling choice either (no tokenization, explicit or implicit) 
    model = eqx_modules.Transformer(
        in_channels = 4,
        out_channels = 4,
        kernel_size = 7, 
        num_layers = 6, 
        n_heads = 4, 
        d_model = 128, 
        d_ff = 64, 
        causal = not MLM,
        key = jr.PRNGKey(0)
    )
elif MODEL == "Charformer": 
    batch_size = 1024 
    # Not having a causal version seems like the biggest weakness to me. 
    assert(MLM)
    model = Charformer(
        input_dim = 4,
        d_model = 128, 
        output_dim = 4,
        blocks = ((1,0),(3,0),(5,0)),
        downsample_factor = 5, 
        num_layers = 6, 
        n_heads = 4, 
        d_ff = 64, 
        key = jr.PRNGKey(0)
    )
elif MODEL == "Convformer": 
    batch_size = 1024 
    model = Convformer(
        input_dim = 4,
        output_dim = 4,
        d_model = 128, 
        downsample_factor = 5, 
        n_heads = 4, 
        d_ff = 64, 
        kernel_size = 7, 
        causal = not MLM,
        gated = False,
        num_layers = 6, 
        final_kernel_size = 7,
        key = jr.PRNGKey(0)
    )
else: 
    raise ValueError(f"Unknown model {MODEL}")
    

#bed_data, genome_dict = epigenome_data.load_data(["GRCg6a"], width = sequence_len) # ["GRCg6a"]

chrom1 = bed_data["chrom"] == "1"

train_data = bed_data[ ~chrom1 ]
test_data = bed_data[ chrom1 ]

train_dataset = epigenome_data.BedDataset(train_data, genome_dict, width = sequence_len, mask = MLM) 
test_dataset = epigenome_data.BedDataset(test_data, genome_dict, width = sequence_len, mask = MLM) 

train_dataloader = jdl.DataLoader(
    train_dataset, # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
    backend='pytorch', # Use 'jax' backend for loading data
    batch_size=batch_size, # Batch size 
    shuffle=True, # Shuffle the dataloader every iteration or not
    drop_last=True, # Drop the last batch or not
    num_workers = num_workers
) # type: ignore

test_dataloader = jdl.DataLoader(
    test_dataset, # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
    backend='pytorch', # Use 'jax' backend for loading data
    batch_size=batch_size, # Batch size 
    shuffle=False, # Shuffle the dataloader every iteration or not
    drop_last=True, # Drop the last batch or not
    num_workers = num_workers
) # type: ignore


num_devices = len(jax.devices())
devices = mesh_utils.create_device_mesh((num_devices,1))
sharding = jshard.PositionalSharding(devices)

rep_sharding = sharding.replicate()

model = eqx.filter_shard(model, rep_sharding)

optim = optax.adam(learning_rate = 3e-3)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

label = "MLM" if MLM else "LM"
results_dir = Path(f"charformer_jax_results/{MODEL}_{label}")
results_dir.mkdir(exist_ok = True, parents = True)

train_losses = []
test_losses = []
for epoch in range(50): 
    # training loop
    start_time = time.time()
    np.random.seed( time.time_ns() % (2**32) )
    model, opt_state, train_loss = loop(train_dataloader, model, rep_sharding, opt_state = opt_state)
    train_losses.append(train_loss)

    # validation loop
    np.random.seed( 0 )
    _, _, test_loss = loop(test_dataloader, model, rep_sharding, opt_state = None)
    test_losses.append(test_loss)

    epoch_time = time.time() - start_time
    print(f"Epoch:{epoch} train_loss:{train_loss:.5} test_loss:{test_loss:.5} took {epoch_time:.2}s")

    eqx.tree_serialise_leaves(results_dir / "checkpoint.pkl", model)
    pd.DataFrame({"train_loss": train_losses, "test_loss" : test_losses}).to_csv(results_dir / "metrics.tsv", sep = "\t", index = False)

import matplotlib.pyplot as plt
metrics = pd.read_csv(results_dir / "metrics.tsv", sep="\t")

plt.plot(metrics["train_loss"], label = "train")
plt.plot(metrics["test_loss"], label = "test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()