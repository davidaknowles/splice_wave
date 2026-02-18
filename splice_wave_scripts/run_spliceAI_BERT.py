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

model = spliceAI.SpliceAI_10k(in_channels = 6, out_channels = 5).to(device)

train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
test_chroms = ["chr1"]

# batch_size = 10. Cad done 2^20 ~ 1M tokens per batch. So og is 10x smaller

train_dataloader = transcript_data.get_dataloader(get_gene, train_chroms, receptive_field = 5000, batch_size = 50, device = device, max_len = 10000 )
test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = 5000, device = device, max_len = 30000 )

optimizer = torch.optim.Adam(model.parameters())

def get_mask(B, T, missing_rate = 0.15, min_span = 30, max_span = 300): 
    bert_mask = torch.zeros( (B, T), dtype = bool, device = device)
    for b in range(B): 
        while bert_mask[b,:].float().mean() < missing_rate: 
            span = np.random.randint(min_span, max_span)
            start = np.random.randint(0, T-span)
            bert_mask[b, start:start + span] = True
    return bert_mask

rf = model.receptive_field

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

        if train: 
            optimizer.zero_grad()
            
        one_hot = one_hot.permute(0, 2, 1) # convert to B x C x T
        is_exon = is_exon.permute(0, 2, 1)
    
        mask = is_exon.isnan() # record what is truly missing in is_exon (because of short genes)
    
        B,C,T = is_exon.shape # batch, channels, length

        # TODO: 80-10-10 mask/corrupt/cheat (for each span)
        meta_mask = get_mask(B, T, min_span = 30, max_span = 300) 
        seq_mask = get_mask(B, T, min_span = 1, max_span = 10)
    
        mask_mask = meta_mask | mask[:,0,:] # masked plus truly missing
    
        meta = torch.zeros( B, 2, T + rf * 2, device = device)  # one hot, zero for missing
        meta[:, 0, rf:-rf][~mask_mask] = is_exon[:,0,:][~mask_mask]
        meta[:, 1, rf:-rf][~mask_mask] = (1.-is_exon[:,0,:])[~mask_mask]
    
        one_hot_masked = one_hot.clone().detach()
        for channel in range(4):
            one_hot_masked[:, channel, rf:-rf][ seq_mask ] = 0. 
    
        input = torch.concat( (meta, one_hot_masked), 1)
        
        output = model(input.nan_to_num()) # spliceAI uses conv which want B x C x T
    
        meta_logits = output[:,0,:]
        
        eval_mask = meta_mask & ~mask[:,0,:] # masked and not actually missing
        is_exon_masked = is_exon[:,0,:][ eval_mask ]
        output_masked = meta_logits[ eval_mask ]
        meta_loss = F.binary_cross_entropy_with_logits(output_masked, is_exon_masked)

        seq_eval_mask = seq_mask & ~mask[:,0,:] 
        seq_out = output[:,1:,:].permute(0, 2, 1)
        seq_out_norm = seq_out - seq_out.logsumexp(2, keepdims = True)
        one_hot_t = one_hot.permute(0, 2, 1)
        seq_loss = - (one_hot_t[:, rf:-rf, :][ seq_eval_mask ] * seq_out_norm[ seq_eval_mask ]).sum() / seq_eval_mask.sum() 
        assert(not seq_loss.isnan().item())
    
        loss = meta_loss + seq_loss
        if train:
            loss.backward()
            optimizer.step()
        losses.append( loss.item() )

        accs.append( (is_exon_masked > 0.5).eq( output_masked > 0. ).float().mean().item() )
        if (time.time() - last_log_time) > 60.0: 
            print("%i %f %f" % (batch_counter, np.mean(losses), np.mean(accs)))
            last_log_time = time.time()
    
    return(np.mean(losses),np.mean(accs))

checkpoint_path = Path("checkpoints_spliceAI_BERT")
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

