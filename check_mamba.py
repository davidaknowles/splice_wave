import os 
import transcript_data
import time
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import glob
import importlib
import tcn
import spliceAI
from pathlib import Path
importlib.reload(tcn)

def get_acc(fn): 
    a=torch.load(fn)
    return(a['train_meta_loss'], a['train_seq_loss'], 1.0 - a["train_acc"], a['test_meta_loss'], a['test_seq_loss'], 1.0 - a["test_acc"])

checkpoint_path = Path("checkpoints_mamba")

n_epoch = len(list(checkpoint_path.glob("*.pt")))

train_meta_loss, train_seq_loss, train_acc, test_meta_loss, test_seq_loss, test_acc = zip(*[ get_acc(checkpoint_path / ("%i.pt" % i)) for i in range(n_epoch) ])

plt.plot(train_acc,"-.o",label="train")
plt.plot(test_acc,":o",label="test")
plt.legend()
#plt.yscale("log")
plt.grid()
plt.grid(which="minor")
#plt.ylim(1e-4,1e-3)

plt.plot(train_meta_loss,"-.o",label="train")
plt.plot(test_meta_loss,":o",label="test")
plt.legend()
#plt.yscale("log")
plt.grid()
plt.grid(which="minor")
#plt.ylim(1e-4,1e-3)
plt.show()

plt.plot(train_seq_loss,"-.o",label="train")
plt.plot(test_seq_loss,":o",label="test")
plt.legend()
#plt.yscale("log")
plt.grid()
plt.grid(which="minor")

get_gene = transcript_data.get_generator(
    os.path.expanduser("~/knowles_lab/index/hg38/hg38.fa.gz"), 
    "gencode.v24.annotation.gtf.gz",
    "ENCFF191YXW.tsv.gz", to_one_hot = False) # neural cell polyA RNA-seq

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = tcn.MambaNet(vocab_size = 5, input_channels = 1, n_embed = 64, n_layers = 8).to(device)

checkpoint = torch.load(checkpoint_path / ("%i.pt" % (n_epoch-1)))
model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = 0, device = device, max_len = 30000, batch_size = 1 )

dataloader = test_dataloader

for ((is_exon, lengths_), (seq, lengths), weights) in dataloader: 
    break

mask = is_exon.isnan()

is_exon_logit = torch.full_like(is_exon, -1.)
        
is_exon_logit.requires_grad = True

optimizer = torch.optim.Adam([is_exon_logit], lr = 0.1)

for i in range(500): 

    is_exon = is_exon_logit.sigmoid()

    seq_out, meta_out = model(seq[:,:-1], is_exon.nan_to_num()[:,:-1,:])

    seq_out_norm = seq_out - seq_out.logsumexp(2, keepdims = True)
    selected_elements = seq_out_norm.gather(2, seq[:,1:].unsqueeze(2))
    seq_loss = -selected_elements[~selected_elements.isnan()].mean()
    assert(not seq_loss.isnan().item())

    meta_out_masked = meta_out[ ~mask[:,1:,:] ]
    is_exon_masked = is_exon[:,1:,:][ ~mask[:,1:,:] ]
    meta_loss = F.binary_cross_entropy_with_logits(meta_out_masked, is_exon_masked)
    assert(not meta_loss.isnan().item())
    loss = seq_loss + meta_loss # 1.2630 for real is_exon 1.2674 with all zero

    loss.backward()
    optimizer.step()

    print(f"{i} {loss.item()}", end = "\r")

stop

plt.figure(figsize = (25,5))
plt.plot(output[2,0,:].sigmoid().detach().cpu().numpy())
plt.plot( is_exon[0,:,0].detach().cpu().numpy())
plt.plot( selected_elements[0,:,0].detach().cpu().numpy())
plt.show()

np.save("gen.npy", is_exon_gen.cpu().numpy())


test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = 0, device = device )
dataloader = test_dataloader

torch.set_grad_enabled()
    
for ((is_exon, lengths_), (seq, lengths), weights) in dataloader: 

    if train: 
        optimizer.zero_grad()

    mask = is_exon.isnan()

    # should really pass "missing" token?
    seq_out, meta_out = model(seq[:,:-1], is_exon.nan_to_num()[:,:-1,:])
    
    #seq_out_unpad = torch.nn.utils.rnn.unpad_sequence(seq_out, lengths - 1, batch_first = True )
    #seq_unpad = torch.nn.utils.rnn.unpad_sequence(seq[:,1:], lengths - 1, batch_first = True )

    seq_out_norm = seq_out - seq_out.logsumexp(2, keepdims = True)
    selected_elements = seq_out_norm.gather(2, seq[:,1:].unsqueeze(2))
    seq_loss = -selected_elements[~selected_elements.isnan()].mean()
    assert(not seq_loss.isnan().item())
    # cross entropy takes Batch x Class x other dims
    #seq_loss = F.cross_entropy(seq_out.permute(0,2,1), seq[:,1:]) # this works but includes stuff that should be masked
    #meta_loss = F.binary_cross_entropy_with_logits(meta_out.nan_to_num(), is_exon[:,1:,:].nan_to_num(), pos_weight = (~mask[:,1:,:]).float() )

    meta_out_masked = meta_out[ ~mask[:,1:,:] ]
    is_exon_masked = is_exon[:,1:,:][ ~mask[:,1:,:] ]
    meta_loss = F.binary_cross_entropy_with_logits(meta_out_masked, is_exon_masked)
    assert(not meta_loss.isnan().item())
    loss = seq_loss + meta_loss
    
    if train:
        loss.backward()
        optimizer.step()
    meta_losses.append( meta_loss.item() )
    seq_losses.append( seq_loss.item() )
