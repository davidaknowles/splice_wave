```python
import os 
import transcript_data
import time
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
import gtf_loader
import utils

tpms = transcript_data.get_tpms("ENCFF191YXW.tsv.gz" )

(exons, genes) = gtf_loader.get_exons("gencode.v24.annotation.gtf.gz")

genome = utils.get_fasta(os.path.expanduser("~/knowles_lab/index/hg38/hg38.fa.gz"))

```

```python
def get_generator(genome, tpms, exons, genes):

    def get_gene(chroms = None, receptive_field=0, max_len = 10000):
        for gene,chrom_strand in genes.items():
            if chroms: 
                if not chrom_strand.chrom in chroms: 
                    continue
            gene_start = np.inf
            gene_end = 0
            for transcript,exons_here in exons[gene].items():
                for exon in exons_here: 
                    gene_start = min(gene_start, exon.start)
                    gene_end = max(gene_end, exon.end)
            transcripts = list(exons[gene].keys())
            num_transcripts = len(transcripts)
            
            if (gene_end - gene_start) > max_len: 
                gene_start = np.random.randint(gene_start, gene_end - max_len)
                gene_end = gene_start + max_len
            
            gene_start -= receptive_field
            gene_end += receptive_field
                
            is_exon = np.zeros((num_transcripts, gene_end - gene_start), dtype=np.float32)

            for transcript_idx,transcript in enumerate(transcripts):
                exons_here = exons[gene][transcript]
                for exon in exons_here: 
                    exon_start = (exon.start-gene_start) if (chrom_strand.strand=="+") else (gene_end - exon.end)
                    exon_end = (exon.end-gene_start) if (chrom_strand.strand=="+") else (gene_end - exon.start)            
                    is_exon[transcript_idx, exon_start:exon_end] = 1

            offset = -1 if (chrom_strand.strand=="+") else 0
            seq = utils.fetch_sequence(genome, chrom_strand.chrom, gene_start + offset, gene_end, chrom_strand.strand)
            # TODO: check seq is valid
            if seq is None: continue
            if len(seq) != (gene_end - gene_start): continue
            start_di = []
            end_di = []
            weights = np.zeros(num_transcripts, dtype=np.float32)
            for transcript_idx,transcript in enumerate(transcripts):
                weights[transcript_idx] = tpms.get(transcript,0.0)
                exons_here = exons[gene][transcript]
                for exon_idx,exon in enumerate(exons_here):
                    if exon_idx > 0 and exon_idx < len(exons_here)-1:
                        exon_start = (exon.start-gene_start) if (chrom_strand.strand=="+") else (gene_end - exon.end)
                        start_di.append( seq[exon_start-2:exon_start] )
                        exon_end = (exon.end-gene_start) if (chrom_strand.strand=="+") else (gene_end - exon.start)
                        end_di.append(seq[exon_end+1:exon_end+3])

            if len(start_di)==0: continue
            start_correct = np.mean( np.array(start_di) == "AG")
            end_correct = np.mean( np.array(end_di) == "GT")

            one_hot = utils.one_hot(seq)
            
            yield(is_exon, one_hot, weights)
            
    return(get_gene)

get_gene = get_generator(genome, tpms, exons, genes)



```

```python
class SpatialDropout1D(nn.Module):

    def __init__(self, p):
        super(SpatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # convert to [batch, channels, time]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)   # back to [batch, time, channels]
        return(x)

class ResBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, stride=1, dropout=0.0):
        super(ResBlock, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout1 = SpatialDropout1D(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout2 = SpatialDropout1D(dropout)

        self.net = nn.Sequential(self.conv1, nn.ELU(), self.dropout1,
                                 self.conv2, nn.ELU(), self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)

        if self.padding == 0: 
            to_trim = (self.kernel_size - 1) * self.dilation
            assert(to_trim % 2 == 0)
            #to_trim /= 2 # don't do this because of two convs
            to_trim = int(to_trim)
            x = x[:,:,to_trim:-to_trim]
    
        if not self.downsample is None: 
            x = self.downsample(x)
        
        return F.relu(out + x)

#%%
class ConvNet(nn.Module):
    def __init__(self, num_channels, kernel_size=2, dropout=0.0):
        super(ConvNet, self).__init__()

        num_levels = len(num_channels)

        self.convs = []

        self.receptive_field = 0
        self.receptive_fields = []

        for i in range(num_levels):
            dilation = 2 ** i

            rf = int( (kernel_size-1) * dilation )
            self.receptive_fields.append(rf)

            assert(rf % 2 == 0)

            channels_in = 6 if i==0 else num_channels[i-1]
            self.convs.append( ResBlock(channels_in, num_channels[i], kernel_size, dilation, 0, dropout=dropout )) # int(padding / 2)
            
        #self.concats.append( keras.layers.Concatenate() )
        self.convs.append( nn.Conv1d(num_channels[num_levels-1], 1, 1, stride=1, padding=0, dilation=dilation) )
        
        self.net = nn.Sequential(*self.convs)
        self.receptive_field = sum(self.receptive_fields)
    
    def forward(self, x):
        return(self.net(x))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ConvNet([32] * 10, kernel_size=7, dropout = 0.2).to(device)
optimizer = torch.optim.Adam(model.parameters())
```


```python


#max_glen = 100000
max_batch = 100

def one_epoch(chroms, train):
    start_time = time.time()
    last_log_time = time.time()
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    losses = []
    accs = []
    gene_counter = 0
    for (is_exon, one_hot, weights) in get_gene(chroms, model.receptive_field, max_len = 10000):
        if weights.sum() == 0: continue
        gene_counter += 1
        to_keep = np.where(weights)[0]
        is_exon = is_exon[to_keep,:]
        weights = weights[to_keep]
        weights = weights / weights.sum()

        one_hot = np.swapaxes(one_hot, 0, 1)

        if is_exon.shape[0] > max_batch: 
            to_keep = np.random.randint(is_exon.shape[0], size=max_batch)
            is_exon = is_exon[to_keep,:]
            weights = weights[to_keep]

        y = is_exon[:,model.receptive_field:-model.receptive_field].copy()[:,np.newaxis,:]
        is_exon[:,model.receptive_field:-model.receptive_field] = 0. 

        x = np.zeros((is_exon.shape[0], 6, is_exon.shape[1]), dtype=np.float32)
        x[:,:4,:] = np.tile(one_hot, (is_exon.shape[0],1,1))
        x[:,4,:] = is_exon
        x[:,5,:] = 1.
        x[:,5,model.receptive_field:-model.receptive_field] = 0. # missing mask

        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        weights = torch.from_numpy(weights[:,np.newaxis,np.newaxis]).to(device)
        if train: 
            optimizer.zero_grad()
        output = model(x)
        loss = F.binary_cross_entropy_with_logits(output, y, weight=weights)
        if train:
            loss.backward()
            optimizer.step()
        losses.append( loss.item() )

        accs.append( (y > 0.5).eq( output > 0. ).float().mean().item() )
        if (time.time() - last_log_time) > 60.0: 
            print("%i %f %f" % (gene_counter, np.mean(losses), np.mean(accs)))
            last_log_time = time.time()

    return(np.mean(losses),np.mean(accs))

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

#n_epoch=0

if False: 
    import glob
    n_epoch = len(glob.glob("checkpoints/*.pt"))
    checkpoint = torch.load("checkpoints/%i.pt" % (n_epoch-1))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for epoch in range(n_epoch, n_epoch + 200): 
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
            }, "checkpoints/%i.pt" % epoch)
```

```python
import glob
n_epoch = len(glob.glob("checkpoints/*.pt"))

def get_acc(fn): 
    a=torch.load(fn)
    return(1.0 - a["train_acc"], 1.0 - a["test_acc"])

accs = [ get_acc("checkpoints/%i.pt" % i) for i in range(n_epoch) ]

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
```

```python
import glob
n_epoch = len(glob.glob("checkpoints_no_dropout/*.pt"))

def get_acc(fn): 
    a=torch.load(fn)
    return(1.0 - a["train_acc"], 1.0 - a["test_acc"])

accs = [ get_acc("checkpoints_no_dropout/%i.pt" % i) for i in range(n_epoch) ]

train_acc_no_dropout, test_acc_no_dropout = zip(*accs)
```

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.plot(train_acc,":o",label="train (do)", color="r")
plt.plot(test_acc,"-o",label="test (do)", color="r")
plt.plot(train_acc_no_dropout,":o",label="train", color="b")
plt.plot(test_acc_no_dropout,"-o",label="test", color="b")
plt.legend()
plt.ylim(1e-4,1e-3)
plt.yscale("log")
plt.show()
```

```python
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