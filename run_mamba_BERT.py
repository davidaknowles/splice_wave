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
importlib.reload(train)


try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError as e:
    print(f"XLA not available, will use GPU or CPU")
    XLA_AVAILABLE = False


pred_meta_task = True

device = xm.xla_device() if XLA_AVAILABLE else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

model = tcn.MambaOneHotNet(in_channels = 6, out_channels = 5, n_embed = 64, n_layers = 8, receptive_field = 5000, bidir = True).to(device)

checkpoint_path = Path("checkpoints_mamba_BERT")
checkpoint_path.mkdir(exist_ok=True)

optimizer = torch.optim.Adam(model.parameters())

if False: # restart from last checkpoint
    import glob
    n_epoch = len(list(checkpoint_path.glob("*.pt")))
    checkpoint = torch.load(checkpoint_path / ("%i.pt" % (n_epoch-1)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# batch_size = 10. Cad done 2^20 ~ 1M tokens per batch. So og is 10x smaller

get_gene = transcript_data.get_generator(
    os.path.expanduser("hg38.fa.gz"), 
    "gencode.v24.annotation.gtf.gz",
    "ENCFF191YXW.tsv.gz") # neural cell polyA RNA-seq

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

train_dataloader = transcript_data.get_dataloader(get_gene, train_chroms, receptive_field = 5000, batch_size = 3, device = device, max_len = 10000 )
test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = 5000, batch_size = 1, device = device, max_len = 30000 )

for epoch in range(100): #  range(n_epoch, n_epoch + 40): 
    np.random.seed(int(time.time()))

    train_metrics = train.one_epoch(model, train_dataloader, optimizer = optimizer, device = device, pred_meta_task = pred_meta_task, eval_LM = False)
    print("TRAIN EPOCH %i complete " % epoch) # TODO fix printing
    print(" ".join( [ "%s:%.4g" % (k,v) for k,v in train_metrics.items() ] ) )

    np.random.seed(1)
    test_metrics = train.one_epoch(model, test_dataloader, optimizer = None, device = device, pred_meta_task = True, eval_LM = True)
    print(" ".join( [ "%s:%.4g" % (k,v) for k,v in test_metrics.items() ] ) )

    to_save = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
    }
    to_save.update(train_metrics)
    to_save.update(test_metrics)
    torch.save(to_save, checkpoint_path / ("%i.pt" % epoch))

