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

import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn

MLM = False

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

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=jax.nn.relu, padding = "SAME", key=None, gated = False):
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

def loop(dataloader, model, rep_sharding, opt_state = None): 
    
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
        
        if step % 500 == 0: 
            print(f"Epoch:{epoch} step:{step}, loss:{loss} rate:{tracker.rate()}")
    
    epoch_loss = np.mean(losses)
    return model, opt_state, epoch_loss

sequence_len = 1024
batch_size = 512
num_workers = 50

bed_data, genome_dict = epigenome_data.load_data(None, width = sequence_len) # ["GRCg6a"]

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

model = FullyConvSeq2Seq(
    in_channels = 4, 
    hidden_channels = 128, 
    out_channels = 4, 
    num_layers = 5, 
    kernel_size = 7, 
    gated = True,
    causal = not MLM,
    key = jr.PRNGKey(0)
)

num_devices = len(jax.devices())
devices = mesh_utils.create_device_mesh((num_devices,1))
sharding = jshard.PositionalSharding(devices)

rep_sharding = sharding.replicate()

model = eqx.filter_shard(model, rep_sharding)

optim = optax.adam(learning_rate = 3e-3)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

results_dir = Path("charformer_jax_results/gated")
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

    eqx.tree_serialise_leaves(results_dir / "charformer_jax.pkl", model)
    pd.DataFrame({"train_loss": train_losses, "test_loss" : test_losses}).to_csv(results_dir / "metrics.tsv", sep = "\t", index = False)