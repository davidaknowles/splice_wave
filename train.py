import torch
import transcript_data
import tcn
import time
import torch.nn.functional as F
import numpy as np

# will probably want a different version for Mamba: everything handled so differently
def one_epoch(model, dataloader, optimizer = None, device = "cpu", pred_meta_task = False, eval_LM = False, max_batches = None):
    rf = model.receptive_field
    
    train = not optimizer is None
    start_time = time.time()
    last_log_time = time.time()
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()

    metrics = []
    
    batch_counter = 0
    for ((is_exon, lengths_), (one_hot, lengths), weights) in dataloader: 

        metrics.append({})
        
        if train: 
            optimizer.zero_grad()

        # convert to B x C x T (CNN) from B x T x C (RNN/transformer)
        # these are generated by rnn.pad_sequence internally
        one_hot = one_hot.permute(0, 2, 1) 
        is_exon = is_exon.permute(0, 2, 1)
    
        mask = is_exon.isnan() # record what is truly missing in is_exon (because of short genes)
    
        B,C,T = is_exon.shape # batch, channels, length

        # TODO: handle multiple meta channels (need to think carefully about joint masking)
        meta = torch.zeros( B, 2, T + rf * 2, device = device)  # one hot, zero for missing
        meta[:, 0, rf:-rf] = is_exon[:,0,:]
        meta[:, 1, rf:-rf] = (1.-is_exon[:,0,:])

        meta_mask = transcript_data.get_mask(B, T, meta, min_span = 30, max_span = 300, mask_same = True) 
    
        one_hot_masked = one_hot.clone().detach()
        seq_mask = transcript_data.get_mask(B, T, one_hot_masked, min_span = 1, max_span = 10, mask_same = False)
    
        input = torch.concat( (meta, one_hot_masked), 1)
        
        output = model(input.nan_to_num()) # spliceAI uses conv which want B x C x T
    
        meta_logits = output[:,0,:]
        
        eval_mask = meta_mask & ~mask[:,0,:] # masked and not actually missing
        is_exon_masked = is_exon[:,0,:][ eval_mask ]
        output_masked = meta_logits[ eval_mask ]
        meta_loss = F.binary_cross_entropy_with_logits(output_masked, is_exon_masked)
        metrics[-1]["meta_acc"] = (is_exon_masked > 0.5).eq( output_masked > 0. ).float().mean().item()
        metrics[-1]["meta_loss"] = meta_loss.item()
        
        seq_loss = tcn.my_bce_loss(seq_mask, mask, output[:,1:,:], one_hot[:, :, rf:-rf]) 
        metrics[-1]["seq_loss"] = meta_loss.item()
    
        loss = meta_loss + seq_loss
        assert(not loss.isnan().item())

        if pred_meta_task: 
            meta = torch.zeros( B, 2, T + rf * 2, device = device)
            input = torch.concat( (meta, one_hot), 1 ) # pred with full sequence but no is_exon info
            output = model(input.nan_to_num()) 
            is_exon_masked = is_exon[:,0,:][ ~mask[:,0,:] ]
            output_masked = output[:,0,:][ ~mask[:,0,:] ]
            loss += F.binary_cross_entropy_with_logits(output_masked, is_exon_masked)
            metrics[-1]["pred_meta_acc"] = (is_exon_masked > 0.5).eq( output_masked > 0. ).float().mean().item()
            metrics[-1]["baseline_acc"] = (is_exon_masked <= 0.5).float().mean().item()
        
        # TODO: could also add an "MLM" task where meta is not masked at all, but my gut is this would be
        # highly redundant with the mask MLM task where both are partially masked
    
        if eval_LM: 
            # evaluate seq LM perf without context
            meta = torch.zeros( B, 2, T + rf * 2, device = device)
            one_hot_masked = one_hot.clone().detach()
            seq_mask = transcript_data.get_mask(B, T, one_hot_masked, cheat_rate = 0., corrupt_rate = 0., min_span = 1, max_span = 10, mask_same = False)
            input = torch.concat( (meta, one_hot_masked), 1)
            output = model(input.nan_to_num()) 
            seq_loss = tcn.my_bce_loss(seq_mask, mask, output[:,1:,:], one_hot[:, :, rf:-rf]) 
            metrics[-1]["no_context"] = seq_loss.item()
        
            # evaluate seq LM perf with context
            meta[:, 0, rf:-rf] = is_exon[:,0,:]
            meta[:, 1, rf:-rf] = (1.-is_exon[:,0,:])
        
            input = torch.concat( (meta, one_hot_masked), 1)
            output = model(input.nan_to_num()) 
        
            seq_loss = tcn.my_bce_loss(seq_mask, mask, output[:,1:,:], one_hot[:, :, rf:-rf]) 
            metrics[-1]["with_context"] = seq_loss.item()

        if train:
            loss.backward()
            optimizer.step()
        
        metrics[-1][ "loss" ] = loss.item()
        metrics[-1][ "time" ] = time.time() - start_time

        if (time.time() - last_log_time) > 60.0: 
            print("%i" % batch_counter, end = '\r')
            last_log_time = time.time()

        batch_counter += 1

        if (not max_batches is None) and (batch_counter >= max_batches): break
    
    keys = list(metrics[0].keys())
    prefix = "train_" if train else "test_"
    return {prefix+key: np.mean([d[key] for d in metrics]) for key in keys}
