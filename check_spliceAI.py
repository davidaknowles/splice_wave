import os 
import transcript_data
import time
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
import glob
import importlib
import tcn
import matplotlib.pyplot as plt

from pathlib import Path
import spliceAI
importlib.reload(tcn)

checkpoint_path = Path("checkpoints_spliceAI")

n_epoch = len(list(checkpoint_path.glob("*.pt")))

def get_acc(fn): 
    a=torch.load(fn)
    return(1.0 - a["train_acc"], 1.0 - a["test_acc"])

accs = [ get_acc(checkpoint_path / ("%i.pt" % i)) for i in range(n_epoch) ]

train_acc, test_acc = zip(*accs)

plt.plot(train_acc,"-.o",label="train")
plt.plot(test_acc,":o",label="test")
plt.legend()
plt.yscale("log")
plt.grid()
plt.grid(which="minor")
#plt.ylim(1e-4,1e-3)
plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = spliceAI.SpliceAI_10k(out_channels = 1).to(device)

checkpoint = torch.load(checkpoint_path / ("%i.pt" % (n_epoch-1)))
model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

get_gene = transcript_data.get_generator(
    os.path.expanduser("~/knowles_lab/index/hg38/hg38.fa.gz"), 
    "gencode.v24.annotation.gtf.gz",
    "ENCFF191YXW.tsv.gz") # neural cell polyA RNA-seq

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = 5000, batch_size = 5, device = device, max_len = 100000 )

for ((is_exon, lengths_), (one_hot, lengths), weights) in test_dataloader: 
    one_hot = one_hot.permute(0, 2, 1)
    mask = is_exon.isnan()
    output = model(one_hot.nan_to_num())
    break    

plt.figure(figsize = (25,5))
plt.plot(output[2,0,:].sigmoid().detach().cpu().numpy())
plt.plot( is_exon[2,:,0].cpu().numpy())
plt.show()


from mamba_ssm import Mamba # wants B x L x C
B = 1
L = 1000
C = 32

m = Mamba( 32 ).to(device)

x = torch.zeros( B, L, C ).to(device)
x[:, 500, 0]= 1. 
y = m(x)

plt.plot( y[0,490:510,0].detach().cpu().numpy())
