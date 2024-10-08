try: 
    import torch_xla
    from torch_xla import runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    XLA_AVAILABLE = True
except ImportError: 
    print("Couldn't import torch_xla") 
    XLA_AVAILABLE = False

import time
import itertools

from pathlib import Path

import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import transcript_data
import spliceAI
import numpy as np
import pandas as pd
import os
import tcn

import time
from utils import RateTracker

class TrainXLADDP():
    
    def __init__(
        self,
        use_xla = False,
        batch_size = 20,
        data_parallel = False,
        down_sample_ratio = 1.,
        sequence_len = 10000,
        num_workers = 0,
        repeats = 1, 
        mamba = False
    ):
    
        self.device = xm.xla_device() if use_xla else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.xla = use_xla

        self.batch_size = batch_size
        self.data_parallel = data_parallel
        self.sequence_len = sequence_len
        
        get_gene = transcript_data.get_generator(
            os.path.expanduser("hg38.fa.gz"), 
            "gencode.v24.annotation.gtf.gz",
            "ENCFF191YXW.tsv.gz",
            down_sample_ratio = down_sample_ratio,
            num_devices = xr.world_size() if data_parallel else 1, 
            device_id = xm.get_ordinal() if data_parallel else 0
        ) # neural cell polyA RNA-seq

        in_channels = 5
        out_channels = 4

        if mamba: 
            self.model = tcn.MambaOneHotNet(
                in_channels = in_channels, 
                out_channels = out_channels, 
                n_embed = 64, 
                n_layers = 8, 
                receptive_field = 5000, 
                bidir = True
            ).to(self.device)
        else: 
            self.model = spliceAI.SpliceAI_10k(
                in_channels = in_channels, 
                out_channels = out_channels, 
                n_embed = 64
            ).to(self.device)

        train_chroms = ["chr%i" % i for i in range(2,23)] + ["chrX"]
        test_chroms = ["chr1"]
        
        # batch_size = 10. Cadaceus done 2^20 ~ 1M tokens per batch. So og is 10x smaller
        train_dataloader = transcript_data.get_dataloader(
            get_gene, 
            train_chroms, 
            receptive_field = 5000, 
            batch_size = self.batch_size, 
            num_workers = num_workers, 
            device = "cpu" if use_xla else self.device, # we pass cpu as the device since MpDeviceLoader handles the transfer to the TPU
            min_len = self.sequence_len, 
            max_len = self.sequence_len,
            repeats = repeats
        )
        self.train_device_loader = pl.MpDeviceLoader(train_dataloader, self.device) if self.xla else train_dataloader

        test_dataloader = transcript_data.get_dataloader(
            get_gene, 
            test_chroms, 
            receptive_field = 5000, 
            batch_size = self.batch_size, 
            num_workers = num_workers, 
            device = "cpu" if use_xla else self.device, 
            min_len = self.sequence_len, 
            max_len = self.sequence_len )
        self.test_device_loader = pl.MpDeviceLoader(test_dataloader, self.device) if self.xla else test_dataloader
        # could use bigger batch here but want to be consistent with mamba
        #test_dataloader = transcript_data.get_dataloader(get_gene, test_chroms, receptive_field = 5000, batch_size = 1, device = "cpu", max_len = 30000 )
        #self.test_device_loader = pl.MpDeviceLoader(test_dataloader, self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def _train_update(self, step, loss, tracker, epoch):
        self.print(f'epoch: {epoch}, step: {step}, loss: {loss}, rate: {tracker.rate()}')
    
    def train_loop_fn(self, train, epoch, verbose = False):

        rf = self.model.receptive_field

        total_loss = torch.tensor(0., device = self.device)
        total_count = 0 # torch.tensor(0., device = self.device, dtype = torch.long)
        
        tracker = RateTracker() 
        if train: 
            self.model.train()

        for step_i, dat in enumerate(self.train_device_loader if train else self.test_device_loader): 

            # we don't actually need lengths so could stop returning? 
            (is_exon, _), (one_hot, lengths), (one_hot_masked, _), (seq_mask, _), weights = dat
            #torch.save([is_exon, one_hot], cache_dir / f"{step_i}.pt")

            # maybe not needed now? 
            #is_exon = is_exon.nan_to_num() # length 10000
            #one_hot = one_hot.nan_to_num() # length 20000

            #if train: 
            self.optimizer.zero_grad()
            
            # convert to B x C x T (CNN) from B x T x C (RNN/transformer)
            # these are generated by rnn.pad_sequence internally
            one_hot = one_hot.permute(0, 2, 1) 
            is_exon = is_exon.permute(0, 2, 1)
            one_hot_masked = one_hot_masked.permute(0, 2, 1)
            # seq_mask = seq_mask.permute(0, 2, 1) # B x T so don't need to do this
            seq_mask_ = seq_mask[:, None, rf:-rf].float() # .expand(-1, 4, -1)
            
            # TODO: handle multiple meta channels (need to think carefully about joint masking)
            meta = F.pad(is_exon, (rf,rf)) # just pad seq dimension

            x = torch.concat( (meta, one_hot_masked), 1)
            
            output = self.model(x) # spliceAI uses conv which needs B x C x T

            one_hot_sub = one_hot[:, :, rf:-rf]
            
            seq_out_norm = output - output.logsumexp(1, keepdims = True)
            loss = - (seq_mask_ * one_hot_sub * seq_out_norm).sum() / (seq_mask_.sum() + 1e-8) 

            if train: 
                loss.backward()
            
            xm.optimizer_step(self.optimizer) if self.data_parallel else self.optimizer.step()
            #if self.xla and not self.data_parallel: 
            #   xm.mark_step()

            total_loss += loss
            total_count += self.batch_size # torch.tensor(self.batch_size, device = self.device)
            
            tracker.add(self.batch_size)

            if step_i % 30 == 0:
                if self.xla:     
                    xm.add_step_closure(self._train_update, args=(step_i, loss, tracker, epoch))
                else: 
                    self._train_update(step_i, loss, tracker, epoch)
        
        if self.xla:     
            self.print("finished")
            xm.add_step_closure(self._train_update, args=(step_i, loss, tracker, epoch))
        else: 
            self._train_update(step_i, loss, tracker, epoch)

        if verbose: self.print("messaged")
        
        if self.xla: 
            
            total_loss = xm.all_reduce(xm.REDUCE_SUM, total_loss)
            if verbose: self.print("reduce done")

            #xm.mark_step()
            #torch_xla.sync()
            if verbose: self.print("markstep done")
            
            #total_loss = total_loss.item()
            total_loss = 0.
            if verbose: self.print("item() done")
        
        if verbose: self.print("results gathered")
        return total_loss, total_count
    
    def master_print(self, *args, **kwargs): 
        xm.master_print(*args, **kwargs) if self.xla else print(*args, **kwargs)
    
    def print(self, *args, **kwargs): 
        if self.xla: args = [f"device {xm.get_ordinal()}"] + args
        print(*args, **kwargs)
    
    def train(self):

        train_losses = []
        test_losses = []
        
        for epoch in range(50):
            if self.xla: 
                #xm.set_rng_state(epoch, device = self.device)
                xm.set_rng_state(epoch, device = self.device)
                #np.random.seed( time.time_ns() % (2**32) )
            loss, total_count = self.train_loop_fn(True, epoch)
            train_losses.append(loss)

            if self.xla: 
                xm.set_rng_state(42, device = self.device)
            test_loss = self.train_loop_fn(False, epoch)
            test_losses.append(test_loss)
            
            self.master_print('Epoch {} train loss {} count {} test loss {} end {}'.format(epoch, loss, total_count, test_loss, time.strftime('%l:%M%p %Z on %b %d, %Y')))

            pd.DataFrame({
                "train_loss" : train_losses, 
                "test_loss" : test_losses}).to_csv("progress.tsv", sep = "\t", index = False)
        if self.xla: 
            xm.wait_device_ops()
        
def _mp_fn(index, flags):
    # "cpu" # xm.xla_device() # == torch_xla.device()
    xla_ddp = TrainXLADDP(**flags)
    xla_ddp.train()

if __name__ == '__main__':

    if False: 
        flags = { 
            "use_xla" : False,
            "batch_size" : 100, 
            "data_parallel" : False,
            "mamba" : True
        } 
    else:
        flags = { 
            "use_xla" : False,
            "batch_size" : 10, 
            "data_parallel" : False,
            "down_sample_ratio" : 1.,
            "repeats" : 1, 
            "mamba" : False
        } 

    print(flags)
    
    if flags["data_parallel"]: 
        xmp.spawn(_mp_fn, args=(flags,))
    else: 
        _mp_fn(0, flags)
