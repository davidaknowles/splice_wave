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
import spliceAI
importlib.reload(transcript_data)

get_gene = transcript_data.get_generator(
    os.path.expanduser("~/knowles_lab/index/hg38/hg38.fa.gz"), 
    "gencode.v24.annotation.gtf.gz",
    "ENCFF191YXW.tsv.gz") # neural cell polyA RNA-seq

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = spliceAI.SpliceAI_10k(out_channels = 1).to(device)

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

train_dataloader = transcript_data.get_dataloader(get_gene, train_chroms, receptive_field = 5000, device = device, max_len = 10000 )
test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = 5000, device = device, max_len = 30000 )

optimizer = torch.optim.Adam(model.parameters())

def one_epoch(dataloader, train):
    start_time = time.time()
    last_log_time = time.time()
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    losses = []
    accs = []
    batch_counter = 0
    for ((is_exon, lengths_), (one_hot, lengths), weights) in dataloader: 
        batch_counter += 1
        one_hot = one_hot.permute(0, 2, 1)

        if train: 
            optimizer.zero_grad()
        
        mask = is_exon.isnan()

        output = model(one_hot.nan_to_num())
        output = output.permute(0, 2, 1)

        is_exon_masked = is_exon[ ~mask ]
        output_masked = output[ ~mask ]
        loss = F.binary_cross_entropy_with_logits(output_masked, is_exon_masked)

        if train:
            loss.backward()
            optimizer.step()
        losses.append( loss.item() )

        accs.append( (is_exon_masked > 0.5).eq( output_masked > 0. ).float().mean().item() )
        if (time.time() - last_log_time) > 60.0: 
            print("%i %f %f" % (batch_counter, np.mean(losses), np.mean(accs)))
            last_log_time = time.time()
    
    return(np.mean(losses),np.mean(accs))

checkpoint_path = Path("checkpoints_spliceAI")
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
    (train_loss,train_acc)=one_epoch(train_dataloader, True)
    print("TRAIN EPOCH %i complete %f %f" % (epoch, train_loss, train_acc))
    np.random.seed(1)
    (test_loss,test_acc)=one_epoch(test_dataloader, False)
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






