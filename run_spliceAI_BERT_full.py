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
import tcn
import matplotlib.pyplot as plt
import train
importlib.reload(tcn)

pred_meta_task = True

get_gene = transcript_data.get_generator(
    os.path.expanduser("~/knowles_lab/index/hg38/hg38.fa.gz"), 
    "gencode.v24.annotation.gtf.gz",
    "ENCFF191YXW.tsv.gz") # neural cell polyA RNA-seq

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = spliceAI.SpliceAI_10k(in_channels = 6, out_channels = 5, n_embed = 64).to(device)

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

# batch_size = 10. Cad done 2^20 ~ 1M tokens per batch. So og is 10x smaller

train_dataloader = transcript_data.get_dataloader(get_gene, train_chroms, receptive_field = 5000, batch_size = 20, device = device, max_len = 10000 )
test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = 5000, device = device, max_len = 30000 )

optimizer = torch.optim.Adam(model.parameters())

checkpoint_path = Path("checkpoints_spliceAI64_BERT_predmeta")
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
    metrics = train.one_epoch(model, train_dataloader, optimizer = optimizer, device = device, pred_meta_task = True, eval_LM = False)
    print("TRAIN EPOCH %i complete %f %f" % (epoch, train_loss, train_acc)) # TODO fix printing
    
    np.random.seed(1)
    metrics = train.one_epoch(model, test_dataloader, optimizer = None, device = device, pred_meta_task = True, eval_LM = True)
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

