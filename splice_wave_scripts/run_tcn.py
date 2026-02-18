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

get_gene = transcript_data.get_generator(
    os.path.expanduser("~/knowles_lab/index/hg38/hg38.fa.gz"), 
    "gencode.v24.annotation.gtf.gz",
    "ENCFF191YXW.tsv.gz") # neural cell polyA RNA-seq

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = tcn.ConvNet([32] * 10, kernel_size=7, dropout = 0.0).to(device)

optimizer = torch.optim.Adam(model.parameters())

max_batch = 10 # max number of transcripts considered

def one_epoch(chroms, train):
    start_time = time.time()
    last_log_time = time.time()
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    losses = []
    accs = []
    gene_counter = 0
    for (is_exon, one_hot, weights) in get_gene(chroms, model.receptive_field, max_len = 30000): 
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
        is_exon = torch.from_numpy(is_exon).to(device)
        one_hot = torch.from_numpy(one_hot).to(device)
        weights = torch.from_numpy(weights[:,np.newaxis,np.newaxis]).to(device)
        
        if train: 
            optimizer.zero_grad()

        output = model(is_exon, one_hot)
        if output.shape[0] < is_exon.shape[0]: 
            output = output.expand(is_exon.shape[0],-1,-1)
        loss = F.binary_cross_entropy_with_logits(output, is_exon, weight=weights)
        if train:
            loss.backward()
            optimizer.step()
        losses.append( loss.item() )

        accs.append( (is_exon > 0.5).eq( output > 0. ).float().mean().item() )
        if (time.time() - last_log_time) > 60.0: 
            print("%i %f %f" % (gene_counter, np.mean(losses), np.mean(accs)))
            last_log_time = time.time()
    
    return(np.mean(losses),np.mean(accs))

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

checkpoint_path = Path("checkpoints_noncausal")
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

