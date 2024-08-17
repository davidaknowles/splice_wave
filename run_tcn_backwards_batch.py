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

args = parser.parse_args([])

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

seq_length = 50000

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

train_dataloader = transcript_data.get_dataloader(get_gene, train_chroms, receptive_field = -model.receptive_field, batch_size = 20, device = device, max_len = seq_length )

# could use bigger batch here but want to be consistent with mamba
test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = -model.receptive_field, batch_size = 20, device = device, max_len = seq_length )

optimizer = torch.optim.Adam(model.parameters())

max_batch = 1 # max number of transcripts considered

def one_epoch(model, dataloader, optimizer = None, device = "cpu", max_batches = None):

    train = not optimizer is None
    
    start_time = time.time()
    last_log_time = time.time()
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    losses = []
    accs = []
    batch_counter = 0
    for ((is_exon, lengths_), (one_hot, lengths), weights) in dataloader:

        one_hot = one_hot.permute(0, 2, 1)
        is_exon = is_exon.permute(0, 2, 1)

        if args.no_exon: 
            is_exon[:] = 0. 
        
        if train: 
            optimizer.zero_grad()

        one_hot_no_nan = one_hot.nan_to_num(0)
        
        output = model(one_hot_no_nan, is_exon.nan_to_num(0))

        output_norm = output - output.logsumexp(1, keepdim = True)
        loss = -(one_hot_no_nan * output_norm).sum(1)[~one_hot[:,0,:].isnan()].mean() # correctly weight all elements
        assert( not loss.isnan().item() )
        
        if train:
            loss.backward()
            optimizer.step()
        losses.append( loss.item() )

        accs.append( (one_hot > 0.5).eq( output > 0. ).float().mean().item() )
        if (time.time() - last_log_time) > 60.0: 
            print("%i %f %f" % (batch_counter, np.mean(losses), np.mean(accs)))
            last_log_time = time.time()

        batch_counter += 1

        if (not max_batches is None) and (batch_counter >= max_batches): break
    
    return(np.mean(losses),np.mean(accs))

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

checkpoint_path = Path("checkpoints_tcn_backwards_batch" + ("" if args.mamba else "_conv") + ("_noexon" if args.no_exon else ""))
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
    (train_loss,train_acc)=one_epoch(model, train_dataloader, optimizer = optimizer, device = device)
    print("TRAIN EPOCH %i complete %f %f" % (epoch, train_loss, train_acc))
    np.random.seed(1)
    (test_loss,test_acc)=one_epoch(model, test_dataloader, optimizer = None, device = device)
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

