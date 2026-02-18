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
importlib.reload(tcn)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mamba", action="store_true")
parser.add_argument("--no_exon", action="store_true")

args = parser.parse_args() # ["--mamba", "--no_exon"]

get_gene = transcript_data.get_generator(
    os.path.expanduser("~/knowles_lab/index/hg38/hg38.fa.gz"), 
    "gencode.v24.annotation.gtf.gz",
    "ENCFF191YXW.tsv.gz") # neural cell polyA RNA-seq

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = tcn.TemporalConvNet(
    [32] * 10, 
    causal_channels = 4, 
    bidir_channels = 1, 
    kernel_size=7, 
    dropout = 0.0,
    mamba = args.mamba
).to(device)

optimizer = torch.optim.Adam(model.parameters())

max_batch = 1 # max number of transcripts considered

def one_epoch(chroms, train):
    start_time = time.time()
    last_log_time = time.time()
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    losses = []
    accs = []
    gene_counter = 0
    for (is_exon, one_hot, weights) in get_gene(chroms, 0, max_len = 50000): 
        if weights.sum() == 0: continue
        gene_counter += 1
        to_keep = np.where(weights)[0]
        is_exon = is_exon[to_keep,:,:]
        weights = weights[to_keep]
        weights = weights / weights.sum()

        is_exon = np.swapaxes(is_exon, 1, 2)
        if is_exon.shape[0] > max_batch: 
            to_keep = np.argsort(-weights)[:max_batch]
            is_exon = is_exon[to_keep,:,:]
            weights = weights[to_keep]

        one_hot = np.swapaxes(one_hot, 0, 1)
        glen = is_exon.shape[-1]

        one_hot = one_hot[np.newaxis,:,:]

        is_exon = F.pad(torch.from_numpy(is_exon), [model.receptive_field,model.receptive_field]).to(device)
        one_hot = torch.from_numpy(one_hot).to(device)
        weights = torch.from_numpy(weights[:,np.newaxis,np.newaxis]).to(device)

        if args.no_exon: 
            is_exon[:] = 0. 
        
        if train: 
            optimizer.zero_grad()

        output = model(one_hot, is_exon) # switched order! 

        output_norm = output - output.logsumexp(1)
        loss = -(one_hot * output_norm).sum(1).mean()
        
        if train:
            loss.backward()
            optimizer.step()
        losses.append( loss.item() )

        accs.append( (one_hot > 0.5).eq( output > 0. ).float().mean().item() )
        if (time.time() - last_log_time) > 60.0: 
            print("%i %f %f" % (gene_counter, np.mean(losses), np.mean(accs)))
            last_log_time = time.time()
    
    return(np.mean(losses),np.mean(accs))

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

checkpoint_path = Path("checkpoints_tcn_backwards" + ("" if args.mamba else "_conv") + ("_noexon" if args.no_exon else ""))
checkpoint_path.mkdir(exist_ok=True)

if False: # restart from last checkpoint
    import glob
    n_epoch = len(glob.glob(checkpoint_path / "*.pt"))
    checkpoint = torch.load(checkpoint_path / ("%i.pt" % (n_epoch-1)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for epoch in range(100): #  range(n_epoch, n_epoch + 40): 
    np.random.seed(int(time.time()))
    (train_loss,train_acc)=one_epoch(train_chroms, True)
    print("TRAIN EPOCH %i complete %f %f" % (epoch, train_loss, train_acc))
    np.random.seed(1)
    (test_loss,test_acc)=one_epoch(test_chroms, False)
    print("TEST EPOCH %i complete %f %f" % (epoch, test_loss, test_acc))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss, 
            'test_loss': test_loss,
            'train_acc': train_acc, 
            'test_acc' : test_acc
            }, checkpoint_path / ("%i.pt" % epoch))

