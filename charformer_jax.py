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

# TODO: residual connections, layernorm
class Conv1DLayer(eqx.Module):
    conv: eqx.nn.Conv1d
    activation: callable

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=jax.nn.relu, key=None):
        self.conv = eqx.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='SAME', key=key)
        self.activation = activation

    def __call__(self, x):
        return self.activation(self.conv(x))

class FullyConvSeq2Seq(eqx.Module):
    layers: list

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, kernel_size=3, stride=1, key=None):

        keys = jr.split(key, num=num_layers + 1)

        self.layers = []
        self.layers.append(Conv1DLayer(in_channels, hidden_channels, kernel_size, stride, key=keys[0]))
        for i in range(1, num_layers):
            self.layers.append(Conv1DLayer(hidden_channels, hidden_channels, kernel_size, stride, key=keys[i]))
        self.layers.append(eqx.nn.Conv1d(hidden_channels, out_channels, kernel_size, stride, padding='SAME', key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    
@eqx.filter_value_and_grad
def compute_loss(model, one_hot_T, one_hot_masked_T, mask):
    output = jax.vmap(model)(one_hot_masked_T)
    out_norm = output - logsumexp(output, axis=1, keepdims=True)
    #seq_mask = mask[:, None, 1:].astype(jnp.float32) # could just sum one_hot_masked_T instead
    seq_mask = mask[:, None, :].astype(jnp.float32) 
    #loss = - (seq_mask * one_hot_T[:, :, 1:] * out_norm[:, :, :-1]).sum() / (seq_mask.sum() + 1e-8)
    loss = - (seq_mask * one_hot_T * out_norm).sum() / (seq_mask.sum() + 1e-8)
    return loss

@eqx.filter_jit
def make_step(model, one_hot_T, one_hot_masked_T, mask, opt_state):
    loss, grads = compute_loss(model, one_hot_T, one_hot_masked_T, mask)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

@eqx.filter_jit(donate="all")
def train_step(model, opt_state, one_hot_T, one_hot_masked_T, mask, rep_sharding):
    #replicated = sharding.replicate()
    model, opt_state = eqx.filter_shard((model, opt_state), rep_sharding)
    #one_hot_T, one_hot_masked_T, mask = eqx.filter_shard((one_hot_T, one_hot_masked_T, mask), replicated)

    loss, grads = compute_loss(model, one_hot_T, one_hot_masked_T, mask)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    model, opt_state = eqx.filter_shard((model, opt_state), rep_sharding)

    return loss, model, opt_state

sequence_len = 1024
batch_size = 512 

bed_data, genome_dict = epigenome_data.load_data(None, width = sequence_len) # ["GRCg6a"]
        
chrom1 = bed_data["chrom"] == "1"

train_data = bed_data[ ~chrom1 ]
test_data = bed_data[ chrom1 ]

train_dataset = epigenome_data.BedDataset(train_data, genome_dict, width = sequence_len, mask = True) 

train_dataset_casted = cast(jdl.datasets.Dataset, train_dataset)

dataloader = jdl.DataLoader(
    train_dataset, # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
    backend='pytorch', # Use 'jax' backend for loading data
    batch_size=batch_size, # Batch size 
    shuffle=True, # Shuffle the dataloader every iteration or not
    drop_last=True, # Drop the last batch or not
    num_workers = 100
) # type: ignore

model = FullyConvSeq2Seq(
    in_channels = 4, 
    hidden_channels = 128, 
    out_channels = 4, 
    num_layers = 5, 
    kernel_size = 7, 
    key = jr.PRNGKey(0)
)

num_devices = len(jax.devices())
devices = mesh_utils.create_device_mesh((num_devices,1))
sharding = jshard.PositionalSharding(devices)

rep_sharding = sharding.replicate()

model = eqx.filter_shard(model, rep_sharding)

opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

#opt_state = jax.device_put_replicated(opt_state, jax.local_devices())
#model_params = jax.device_put_replicated(eqx.filter(model, eqx.is_inexact_array), jax.local_devices())
#train_key = jax.device_put_replicated(train_key, jax.local_devices())

epoch_losses = []
for epoch in range(50): 
    losses = []
    tracker = RateTracker()
    
    for step, batch in enumerate(dataloader):
        species, tissue, assay, one_hot, one_hot_masked, mask = batch
        
        one_hot_masked_T = np.swapaxes(one_hot_masked, 1, 2)
        one_hot_T = np.swapaxes(one_hot, 1, 2)
    
        one_hot, one_hot_masked, mask = eqx.filter_shard((one_hot, one_hot_masked, mask), rep_sharding)
        
        loss, model, opt_state = train_step(model, opt_state, one_hot_T, one_hot_masked_T, mask, rep_sharding)
        loss = loss.item()
        losses.append(loss)

        tracker.add(batch_size)
        
        if step % 500 == 0: 
            print(f"Epoch:{epoch} step:{step}, loss:{loss} rate:{tracker.rate()}")
    
    epoch_loss = np.mean(losses)
    epoch_losses.append(epoch_loss)
    print(f"Epoch:{epoch} loss:{epoch_loss}")