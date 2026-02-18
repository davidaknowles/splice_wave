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
importlib.reload(tcn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = tcn.ConvNet([32] * 10, kernel_size=7, dropout = 0.2).to(device)

n_epoch = len(glob.glob("checkpoints_noncausal/*.pt"))
checkpoint = torch.load("checkpoints_noncausal/%i.pt" % (n_epoch-1))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

get_gene = transcript_data.get_generator(
    os.path.expanduser("hg38.fa.gz"), 
    "gencode.v24.annotation.gtf.gz",
    "ENCFF191YXW.tsv.gz") # neural cell polyA RNA-seq

for (is_exon, one_hot, weights) in get_gene(["chr1"], model.receptive_field, max_len = 100000): 
    if weights.sum() == 0: continue
    to_keep = np.where(weights)[0]
    is_exon = is_exon[to_keep,:,:]
    weights = weights[to_keep]
    weights = weights / weights.sum()
    
    is_exon = np.swapaxes(is_exon, 1, 2)

    one_hot = np.swapaxes(one_hot, 0, 1)
    glen = is_exon.shape[-1]

    one_hot = one_hot[np.newaxis,:,:]
    is_exon = torch.from_numpy(is_exon).to(device)
    one_hot = torch.from_numpy(one_hot).to(device)
    break 

output = model(is_exon, one_hot)

plt.plot(output[0,0,:].sigmoid().detach().cpu().numpy())
plt.plot( is_exon[0,0,:].cpu().numpy())
plt.show()

np.save("gen.npy", is_exon_gen.cpu().numpy())

def get_acc(fn): 
    a=torch.load(fn)
    return(1.0 - a["train_acc"], 1.0 - a["test_acc"])

accs = [ get_acc("checkpoints_noncausal/%i.pt" % i) for i in range(n_epoch) ]

train_acc, test_acc = zip(*accs)

import matplotlib.pyplot as plt
plt.plot(train_acc,"-.o",label="train")
plt.plot(test_acc,":o",label="test")
plt.legend()
plt.yscale("log")
plt.grid()
plt.grid(which="minor")
#plt.ylim(1e-4,1e-3)
plt.show()

for (is_exon, one_hot, weights) in get_gene(test_chroms, model.receptive_field, max_len = 100000): 
    if weights.sum() == 0.: continue
    if is_exon.shape[1] > 4000: continue
    break
```

```python
plt.plot(is_exon[0,:,0])
plt.plot(is_exon[1,:,0], ":")
plt.plot(is_exon[2,:,0], "--")
plt.show()
```

```python
ie_np=is_exon[:,:,0].copy()
is_exon[:,:,:]=0.
#np.where( np.logical_and( ie[:-1] == 1. , ie[1:] == 0. ) )
```

```python
to_keep = np.where(weights)[0]
is_exon = is_exon[to_keep,:,:]
weights = weights[to_keep]
weights = weights / weights.sum()

is_exon = np.swapaxes(is_exon, 1, 2)
if is_exon.shape[0] > max_batch: 
    to_keep = np.random.randint(is_exon.shape[0], size=max_batch)
    is_exon = is_exon[to_keep,:,:]
    weights = weights[to_keep]

one_hot = np.swapaxes(one_hot, 0, 1)
one_hot = one_hot[np.newaxis,:,:]
is_exon = torch.from_numpy(is_exon).to(device)
one_hot = torch.from_numpy(one_hot).to(device)
weights = torch.from_numpy(weights[:,np.newaxis,np.newaxis]).to(device)
model.eval()
torch.set_grad_enabled(False)
output = model(is_exon, one_hot)

output_np = output.cpu().numpy()
```

```python
output_np=output_np[0,:,:].squeeze()
plt.plot(output_np)
#ie=is_exon[0,:,0]
plt.plot(ie)
plt.show()
```

```python
def softplus(x, limit=30.):
    #return(x if x>limit else np.log1p(np.exp(x)))
    return(np.log1p(np.exp(x)))
def log_logistic(x):
    return(-softplus(-x))
```

```python
ie = ie_np[0,:]
state = ie[0]
log_prob_no_change = 0.
prob_change = np.zeros_like(ie)
for i in range(1,len(ie)):
    new_state = ie[i]
    if state==1. and new_state==0.:# 5'SS, exon to intron
        log_prob_no_change = 0.
    log_prob_change_here = log_logistic(output_np[i])
    prob_change[i] = log_prob_no_change + log_prob_change_here
    log_prob_no_change += -output_np[i] + log_prob_change_here
    #log_prob_no_change += np.log(1.-np.exp(log_prob_change_here))
    state = new_state
plt.plot(prob_change)
plt.plot(ie)
plt.show()
```


```python
plt.plot(log_logistic(output_np))
plt.plot(ie)
```