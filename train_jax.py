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
import charformer_jax
import equinox.nn as nn
import mamba_jax
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('model', type=str, help='Conv, Charformer, Transformer or Convformer')

parser.add_argument('-g', '--genome_set', type=str, default = "all", help="all, small or a specific genome")

parser.add_argument('-m', '--mlm', action='store_true', help='Masked language modeling rather than autoregressive')

#args = parser.parse_args(['Mamba','-m','-g','GRCg6a'])
args = parser.parse_args()

#@eqx.filter_value_and_grad
def compute_loss(model, data):
    if args.mlm: 
        one_hot_T, x, mask = data
    else: 
        x = data
    output = jax.vmap(model)(x)
    out_norm = output - logsumexp(output, axis=1, keepdims=True)
    #seq_mask = mask[:, None, 1:].astype(jnp.float32) # could just sum one_hot_masked_T instead
    if args.mlm: 
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
        if args.mlm: 
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

if args.model == "Conv": 
    batch_size = 1024 
    # fully convolutional network, no transformers or similar so limited receptive field
    model = eqx_modules.FullyConvSeq2Seq(
        in_channels = 4, 
        hidden_channels = 128, 
        out_channels = 4, 
        num_layers = 5, 
        kernel_size = 7, 
        gated = True,
        causal = not args.mlm,
        key = jr.PRNGKey(0)
    )
elif args.model == "Transformer": 
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
        causal = not args.mlm,
        key = jr.PRNGKey(0)
    )
elif args.model == "Charformer": 
    batch_size = 1024 
    # Not having a causal version seems like the biggest weakness to me. 
    assert(args.mlm)
    model = charformer_jax.Charformer(
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
elif args.model == "Convformer": 
    batch_size = 1024 
    model = charformer_jax.Convformer(
        input_dim = 4,
        output_dim = 4,
        d_model = 128, 
        downsample_factor = 5, 
        n_heads = 4, 
        d_ff = 64, 
        kernel_size = 7, 
        causal = not args.mlm,
        gated = False,
        num_layers = 6, 
        final_kernel_size = 7,
        key = jr.PRNGKey(0)
    )
elif args.model == "Mamba": 
    batch_size = 64
    if args.mlm: 
        print("Warning: Mamba is an odd choice for MLM, bidirectional Mamba would make more sense")
    model = mamba_jax.Mamba(
        in_channels = 4,
        out_channels = 4, 
        kernel_size = 7, 
        num_layers = 4,
        d_model = 64,
        key = jr.PRNGKey(0)
    )
else: 
    raise ValueError(f"Unknown model {args.model}")

if args.genome_set == "all": 
    genome_set = None
elif args.genome_set == "small":
    genome_set = ["galGal5", "Xenopus_tropicalis_v9.1", "ARS1", "GRCm38", "GRCg6a"]
else: 
    genome_set = [args.genome_set] # this should be a single genome, e.g. GRCg6a
#bed_data, genome_dict = epigenome_data.load_data(genome_set, width = sequence_len) # ["GRCg6a"]

chrom1 = bed_data["chrom"] == "1"

train_data = bed_data[ ~chrom1 ]
test_data = bed_data[ chrom1 ]

train_dataset = epigenome_data.BedDataset(train_data, genome_dict, width = sequence_len, mask = args.mlm) 
test_dataset = epigenome_data.BedDataset(test_data, genome_dict, width = sequence_len, mask = args.mlm) 

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

label = "MLM" if args.mlm else "LM"
results_dir = Path(f"jax_results/{args.model}_{label}_{args.genome_set}")
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

basedir = Path("jax_results")
for results_dir in basedir.glob("*"): 
    metrics = pd.read_csv(results_dir / "metrics.tsv", sep="\t")
    name = results_dir.name
    plt.plot(metrics["train_loss"], label = f"{name}_train")
    plt.plot(metrics["test_loss"], label = f"{name}_test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()