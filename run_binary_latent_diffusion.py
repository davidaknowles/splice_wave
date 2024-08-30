import torch

import denoising_diffusion_1d
from denoising_diffusion_1d import Unet1D, GaussianDiffusion1D

import sys
import os 
import transcript_data
import time
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import importlib
import tcn
import matplotlib.pyplot as plt
import train



import binary_latent_diffusion
importlib.reload(binary_latent_diffusion)
# batch x channels x seq_length
# features are normalized from 0 to 1

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("objective", choices=["pred_noise", "pred_x0", "pred_v"], help="Objective to optimize")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
parser.add_argument("--onehot", action="store_true", help="One hot rather than 2 bit representation of sequence")

args = parser.parse_args()
#args = parser.parse_args(["pred_noise", "--onehot"])

get_gene = transcript_data.get_generator(
    os.path.expanduser("~/knowles_lab/index/hg38/hg38.fa.gz"), 
    "gencode.v24.annotation.gtf.gz",
    "ENCFF191YXW.tsv.gz") # neural cell polyA RNA-seq

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seq_length = 16384 # 8192

unet = Unet1D(
    dim = 32,
    dim_mults = (1, 2, 4, 4, 4, 4, 8),
    channels = 5 if args.onehot else 3 
)

model = binary_latent_diffusion.BinaryLatentDiffusion(
    unet,
    seq_length = seq_length, 
    objective = args.objective
).to(device)

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

# batch_size = 10. Cadaceus done 2^20 ~ 1M tokens per batch. So og is 10x smaller
train_dataloader = transcript_data.get_dataloader(get_gene, train_chroms, receptive_field = 0, batch_size = 20, device = device, max_len = seq_length )

# could use bigger batch here but want to be consistent with mamba
test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = 0, batch_size = 20, device = device, max_len = seq_length )

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

checkpoint_path = Path(f"checkpoints_bld/{args.objective}_{args.lr}_onehot{args.onehot}")
checkpoint_path.mkdir(parents=True, exist_ok=True)

# will probably want a different version for Mamba: everything handled so differently
def one_epoch(model, dataloader, optimizer = None, device = "cpu", max_batches = None):
    #rf = model.receptive_field
    
    train = not optimizer is None
    start_time = time.time()
    last_log_time = time.time()
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()

    metrics = []
    
    batch_counter = 0
    for ((is_exon, lengths_), (one_hot, lengths), weights) in dataloader: 

        metrics.append({})
        
        if train: 
            optimizer.zero_grad()

        # convert to B x C x T (CNN) from B x T x C (RNN/transformer)
        # these are generated by rnn.pad_sequence internally
        one_hot = one_hot.permute(0, 2, 1)
        is_exon = is_exon.permute(0, 2, 1)

        B,C,T = is_exon.shape # batch, channels, length

        if args.onehot: 
            binary_seq = one_hot
        else: 
            one_hot = one_hot.bool() 
            binary_seq = torch.zeros(B, 2, T, device = device) 
            binary_seq[:, 0] = one_hot[:, 1, :] | one_hot[:, 3, :]
            binary_seq[:, 1] = one_hot[:, 2, :] | one_hot[:, 3, :]
    
        mask = is_exon.isnan() # record what is truly missing in is_exon (because of short genes)
            
        input = torch.concat( (is_exon, binary_seq), 1)
        if T < seq_length: # more succinct padding? 
            input = torch.concat( (input, torch.zeros(B,unet.channels,seq_length-T, device = device)), 2)

        model_out = model(input.nan_to_num())

        if train:
            model_out["loss"].backward()
            optimizer.step()
        
        metrics[-1][ "loss" ] = model_out["loss"].item()
        metrics[-1][ "acc" ] = model_out["acc"].item()
        
        metrics[-1][ "time" ] = time.time() - start_time

        if (time.time() - last_log_time) > 60.0: 
            print("%i %s" % (batch_counter, " ".join( [ "%s:%.4g" % (k,v) for k,v in metrics[-1].items() ] )), end = '\n')
            last_log_time = time.time()

        batch_counter += 1

        if (not max_batches is None) and (batch_counter >= max_batches): break
    
    keys = list(metrics[0].keys())
    prefix = "train_" if train else "test_"
    return {prefix+key: np.mean([d[key] for d in metrics]) for key in keys}, {prefix+key: np.median([d[key] for d in metrics]) for key in keys}

if False: # restart from last checkpoint
    import glob
    n_epoch = len(glob.glob(checkpoint_path / "*.pt"))
    checkpoint = torch.load(checkpoint_path / ("%i.pt" % (n_epoch-1)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for epoch in range(100): #  range(n_epoch, n_epoch + 40): 
    np.random.seed(int(time.time()))

    train_metrics, train_metrics_median = one_epoch(model, train_dataloader, optimizer = optimizer, device = device)
    print("TRAIN EPOCH %i complete " % epoch) # TODO fix printing
    print(" ".join( [ "%s:%.4g" % (k,v) for k,v in train_metrics.items() ] ) )

    if train_metrics_median['train_loss'] > 100.: 
        print("Train loss too high, exiting") 
        sys.exit()
    
    np.random.seed(1)
    test_metrics, _ = one_epoch(model, test_dataloader, optimizer = None, device = device)
    print(" ".join( [ "%s:%.4g" % (k,v) for k,v in test_metrics.items() ] ) )
    to_save = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
    }
    to_save.update(train_metrics)
    to_save.update(test_metrics)
    torch.save(to_save, checkpoint_path / ("%i.pt" % epoch))