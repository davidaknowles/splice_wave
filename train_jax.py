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
import mamba_tpu
import wiki_data
import argparse
import recurrentgemma

#jax.config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()

parser.add_argument('model', type=str, help='Mamba, BidirMamba, Conv, Charformer, Transformer or Convformer')

parser.add_argument('-g', '--genome_set', type=str, default = "all", help="all, small or a specific genome e.g. GRCg6a")

parser.add_argument('-m', '--mlm', action='store_true', help='Masked language modeling rather than autoregressive')

parser.add_argument('-f', '--norm_last', action='store_true', help='Only relevant for Mamba')
parser.add_argument('-l', '--layer_norm', action='store_true', help='Use LayerNorm instead of RMSNorm. Only relevant for Mamba')

parser.add_argument('-c', '--context', action='store_true', help='Use context (species, tissues, assay). Only relevant for Mamba models')

parser.add_argument('-i', '--inject', action='store_true', help='Use context (species, tissues, assay) at every position, not just h0.')

#args = parser.parse_args(['BidirRG','-g','GRCg6a','-c'])
args = parser.parse_args()

print(args)

#@eqx.filter_value_and_grad
def compute_loss(model, data):
    if args.mlm: 
        species, tissue, assay, one_hot_T, x, mask = data
    else: 
        species, tissue, assay, x = data
    #output = jax.vmap(model)(x) # do this outside now to accommodate models like Mamba that aleady handle batch
    if args.context:
        output = model(x, context = [species, tissue, assay])
    else: 
        output = model(x)
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

def loop(dataloader, model, rep_sharding = None, opt_state = None, print_every = 10): 

    start_time = time.time()
    
    losses = []
    tracker = RateTracker()
    
    for step, batch in enumerate(dataloader):
        if args.mlm: 
            species, tissue, assay, one_hot, one_hot_masked, mask = batch
            one_hot_masked_T = np.swapaxes(one_hot_masked, 1, 2)
            one_hot_T = np.swapaxes(one_hot, 1, 2)
            data = (species, tissue, assay, one_hot_T, one_hot_masked_T, mask)
        else: 
            species, tissue, assay, one_hot = batch
            data = [species, tissue, assay, np.swapaxes(one_hot, 1, 2)]

        if rep_sharding is not None: 
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

num_devices = len(jax.devices())
devices = mesh_utils.create_device_mesh((num_devices,1))
sharding = jshard.PositionalSharding(devices)
rep_sharding = sharding.replicate()

if args.genome_set == "wiki": 
    n_channels = 42
    data = wiki_data.load_dataset("wikipedia", "20220301.en", split='train', trust_remote_code=True)

    train_test_split = data.train_test_split(test_size=0.2)

    train_dataset = wiki_data.WikiDataset(train_test_split['train'], mask = args.mlm)
    test_dataset = wiki_data.WikiDataset(train_test_split['test'], mask = args.mlm)
else: 
    n_channels = 4
    if args.genome_set == "all": 
        genome_set = None
    elif args.genome_set == "small":
        genome_set = ["galGal5", "Xenopus_tropicalis_v9.1", "ARS1", "GRCm38", "GRCg6a"]
    else: 
        genome_set = [args.genome_set] # this should be a single genome, e.g. GRCg6a
    bed_data, genome_dict = epigenome_data.load_data(genome_set, width = sequence_len) # ["GRCg6a"]
    
    chrom1 = bed_data["chrom"] == "1"
    
    train_data = bed_data[ ~chrom1 ]
    test_data = bed_data[ chrom1 ]

    context_dims = [
        len(bed_data.species.cat.categories), 
        len(bed_data.tissue.cat.categories), 
        len(bed_data.assay.cat.categories)
    ]
                                             
    train_dataset = epigenome_data.BedDataset(train_data, genome_dict, width = sequence_len, mask = args.mlm) 
    test_dataset = epigenome_data.BedDataset(test_data, genome_dict, width = sequence_len, mask = args.mlm) 


model_name = args.model

if args.model == "Conv": 
    batch_size = 1024 
    # fully convolutional network, no transformers or similar so limited receptive field
    model = eqx_modules.FullyConvSeq2Seq(
        in_channels = n_channels, 
        hidden_channels = 128, 
        out_channels = n_channels, 
        num_layers = 5, 
        kernel_size = 7, 
        gated = True,
        causal = not args.mlm,
        key = jr.PRNGKey(0)
    )
    model = jax.vmap(model)
elif args.model == "Transformer": 
    batch_size = 128
    # convolutional input and output layers, and transformer (with RoPE) stack inbetween _at full nucleotide resolution_ which is likely inefficient computationally and not a great modeling choice either (no tokenization, explicit or implicit) 
    model = eqx_modules.Transformer(
        in_channels = n_channels,
        out_channels = n_channels,
        kernel_size = 7, 
        num_layers = 6, 
        n_heads = 4, 
        d_model = 128, 
        d_ff = 64, 
        causal = not args.mlm,
        key = jr.PRNGKey(0)
    )
    model = jax.vmap(model)
elif args.model == "Charformer": 
    batch_size = 1024 
    # Not having a causal version seems like the biggest weakness to me. 
    assert(args.mlm)
    model = charformer_jax.Charformer(
        input_dim = n_channels,
        d_model = 128, 
        output_dim = n_channels,
        blocks = ((1,0),(3,0),(5,0)),
        downsample_factor = 5, 
        num_layers = 6, 
        n_heads = 4, 
        d_ff = 64, 
        key = jr.PRNGKey(0)
    )
    model = jax.vmap(model)
elif args.model == "Convformer": 
    batch_size = 1024 
    model = charformer_jax.Convformer(
        input_dim = n_channels,
        output_dim = n_channels,
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
    model = jax.vmap(model)
elif args.model in ["Mamba", "BidirMamba"]: 
    # setup for mamba is a bit different because we don't vmap the model - it already handles a batch
    # dimension (because the underlying pallas scan does) 
    batch_size = 128 # 128 with pallas, 32 native scan (about 8x slower), 16 with associative scan and SUPER slow
    from jax.sharding import Mesh, PartitionSpec as P
    mesh = Mesh(devices, axis_names=('i', 'j')) # 4x1 so no point using j? 
    shard_map_kwargs = { # works but maybe could be optimized? 
        "mesh" : mesh,
        "in_specs" : (P("i",None,None),P("i",None,None),P("i",None)), 
        "out_specs" : (P("i",None,None),P("i",None)),
        "check_rep" : False
    }
    if args.mlm and args.model=="Mamba": 
        print("Warning: Mamba is an odd choice for MLM, bidirectional Mamba would make more sense")
    model = mamba_tpu.MambaModel(
        in_channels = n_channels,
        out_channels = n_channels, 
        kernel_size = 7, 
        num_layers = 6, 
        d_model = 32, # really slow if we make this bigger since SSM state is 2*d_model^2
        bidir = args.model == "BidirMamba", 
        norm_last = args.norm_last, 
        layer_norm = args.layer_norm, 
        context_dims = context_dims if args.context else [],
        inject = args.inject,
        shard_map_kwargs = shard_map_kwargs,
        key = jr.PRNGKey(0)
    )
    model_name = ("inject-" if args.inject else "") + ("context-" if args.context else "") + args.model + ("-normlast" if args.norm_last else "") + ("-layernorm" if args.layer_norm else "")
elif args.model in ["RG", "BidirRG"]: 
    # setup for mamba is a bit different because we don't vmap the model - it already handles a batch
    # dimension (because the underlying pallas scan does) 
    batch_size = 512 # 128 with pallas, 32 native scan (about 8x slower), 16 with associative scan and SUPER slow
    from jax.sharding import Mesh, PartitionSpec as P
    mesh = Mesh(devices, axis_names=('i', 'j')) # 4x1 so no point using j? 
    shard_map_kwargs = { # works but maybe could be optimized? 
        "mesh" : mesh,
        "in_specs" : (P("i",None,None),P("i",None,None),P("i",None)), 
        "out_specs" : (P("i",None,None),P("i",None)),
        "check_rep" : False
    }
    if args.mlm and args.model=="RG": 
        print("Warning: RG is an odd choice for MLM, BidirRG would make more sense")
    model = recurrentgemma.RecurrentGemmaModel(
        in_channels = n_channels,
        out_channels = n_channels, 
        kernel_size = 7, 
        num_layers = 6, 
        num_heads = 4,
        d_model = 128, 
        bidir = args.model == "BidirRG", 
        context_dims = context_dims if args.context else [],
        shard_map_kwargs = shard_map_kwargs,
        key = jr.PRNGKey(0)
    )
    model_name = ("context-" if args.context else "") + args.model
else: 
    raise ValueError(f"Unknown model {args.model}")

train_dataloader = jdl.DataLoader(
    train_dataset, # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
    backend='pytorch', 
    batch_size=batch_size, # Batch size 
    shuffle=True, # Shuffle the dataloader every iteration or not
    drop_last=True, # Drop the last batch or not
    num_workers = num_workers
) # type: ignore

test_dataloader = jdl.DataLoader(
    test_dataset, # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
    backend='pytorch', 
    batch_size=batch_size, # Batch size 
    shuffle=False, # Shuffle the dataloader every iteration or not
    drop_last=True, # Drop the last batch or not
    num_workers = num_workers
) # type: ignore

model = eqx.filter_shard(model, rep_sharding)

sched = optax.warmup_cosine_decay_schedule(
    init_value = 1e-6, 
    peak_value = 1e-3, 
    warmup_steps = 10000, 
    decay_steps = 20000 * 50, 
    end_value = 1e-4
)
optim = optax.adam(learning_rate = 3e-3)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

label = "MLM" if args.mlm else "LM"
results_dir = Path(f"jax_results/{model_name}_{label}_{args.genome_set}")
results_dir.mkdir(exist_ok = True, parents = True)

train_losses = []
test_losses = []

patience = 5
patience_counter = patience
best_val_loss = np.inf

checkpoint_file = results_dir / "checkpoint.pkl"
metrics_file = results_dir / "metrics.tsv"
if checkpoint_file.exists():
    model = eqx.tree_deserialise_leaves(checkpoint_file, model) 
    df = pd.read_csv(metrics_file, sep="\t")
    train_losses = df['train_loss'].tolist()
    test_losses = df['test_loss'].tolist()
    best_val_loss = np.min(test_losses)

for epoch in range(30): 
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
    print(f"Epoch:{epoch} train_loss:{train_loss:.5} test_loss:{test_loss:.5} took {epoch_time:.2}s patience {patience_counter}")
    pd.DataFrame({"train_loss": train_losses, "test_loss" : test_losses}).to_csv(metrics_file, sep = "\t", index = False)

    if test_loss < best_val_loss: # only checkpoint best
        eqx.tree_serialise_leaves(checkpoint_file, model)
        best_val_loss = test_loss
        patience_counter = patience
    else:
        patience_counter -= 1
        if patience_counter <= 0:
            #model = eqx.tree_deserialise_leaves(results_dir / "checkpoint.pkl", model_original) # this would only be useful if we were doing something else with model afterward
            break
    


import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# MLM wiki: Mamba >> everything (no Charformer res?) 
# LM wiki: mamba >> everything. no charformer because it can't do LM
# LM small: mamba > conv > convformer > transformer
# MLM small: transformer > conv/mamba > conformer/charformer. (although transformer is unstable at end?)

# TODO: 
# Transformer, Charformer MLM wiki
# mamba MLM small
line_styles = ['--', ':', '-.', ':', '--', ':', '-.']
basedir = Path("jax_results")
for i,results_dir in enumerate(basedir.glob("*Ma*_LM_small")): 
    fn = results_dir / "metrics.tsv"
    if not fn.exists(): 
        continue
    metrics = pd.read_csv(fn, sep="\t")
    name = results_dir.name
    #plt.plot(metrics["train_loss"], label = f"{name}_train")
    plt.plot(metrics["test_loss"], linestyle=line_styles[i], label = f"{name}_test", alpha = 0.6)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
plt.legend()