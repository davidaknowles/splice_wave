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
import collections
import tcn
importlib.reload(transcript_data)

get_gene = transcript_data.get_generator(
    os.path.expanduser("~/knowles_lab/index/hg38/hg38.fa.gz"), 
    "gencode.v24.annotation.gtf.gz",
    "ENCFF191YXW.tsv.gz", # neural cell polyA RNA-seq
    to_one_hot = False
)

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# cadaceus used n_embed = 256, n_layers = 8 for their 32K sequence length model
model = tcn.MambaNet(vocab_size = 5, input_channels = 1, n_embed = 64, n_layers = 8).to(device)

train_dataloader = transcript_data.get_dataloader(get_gene, train_chroms, receptive_field = 0, device = device )
test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = 0, device = device )

optimizer = torch.optim.Adam(model.parameters())

def one_epoch(dataloader, train):
    start_time = time.time()
    last_log_time = time.time()
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    meta_losses = []
    seq_losses = []
    accs = []
    gene_counter = 0
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

        accs.append( (is_exon_masked > 0.5).eq( meta_out_masked > 0. ).float().mean().item() )
        if (time.time() - last_log_time) > 60.0: 
            print("%i %.3f %.3f %f" % (gene_counter, np.mean(meta_losses), np.mean(seq_losses), 1.-np.mean(accs)))
            last_log_time = time.time()
    
    return(np.mean(meta_losses),np.mean(seq_losses),np.mean(accs))

checkpoint_path = Path("checkpoints_mamba")
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
    (train_meta_loss,train_seq_loss,train_acc)=one_epoch(train_dataloader, True)
    print("TRAIN EPOCH %i complete %f %f %f" % (epoch, train_meta_loss, train_seq_loss, train_acc))
    np.random.seed(1)
    (test_meta_loss,test_seq_loss,test_acc)=one_epoch(test_dataloader, False)
    print("TEST EPOCH %i complete %f %f %f" % (epoch, test_meta_loss, test_seq_loss, test_acc))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_meta_loss': train_meta_loss, 
            'train_seq_loss': train_seq_loss, 
            'test_meta_loss': test_meta_loss, 
            'test_seq_loss': test_seq_loss, 
            'train_acc': train_acc, 
            'test_acc' : test_acc
            }, checkpoint_path / ("%i.pt" % epoch))

