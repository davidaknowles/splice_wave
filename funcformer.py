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
import importlib
import torch.utils.data 

import epigenome_data
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
        sequence_len = 1024,
        num_workers = 0, 
        bidir = False
    ):
    
        self.device = xm.xla_device() if use_xla else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.xla = use_xla

        self.batch_size = batch_size
        self.data_parallel = data_parallel
        self.sequence_len = sequence_len

        train_dataset, test_dataset, bed_data = epigenome_data.load_data(["ARS1"], width = sequence_len)

        #tissue_embeds = pd.read_csv(vertebrate_epigenomes / "tissue_embeds.tsv", sep = "\t", index_col = 0)
        
        in_channels = 4
        out_channels = 4

        self.model = tcn.MambaOneHotNet(
            in_channels = in_channels, 
            out_channels = out_channels, 
            n_embed = 64, 
            n_layers = 8, 
            receptive_field = 0, 
            L = sequence_len, 
            batch_size = batch_size,
            bidir = bidir
        ).to(self.device)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False) 
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True) 

        self.train_device_loader = pl.MpDeviceLoader(train_dataloader, self.device) if self.xla else train_dataloader

        self.test_device_loader = pl.MpDeviceLoader(test_dataloader, self.device) if self.xla else test_dataloader
        
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def _train_update(self, step, loss, tracker, epoch):
        self.print(f'epoch: {epoch}, step: {step}, loss: {loss}, rate: {tracker.rate()}')
    
    def train_loop_fn(self, train, epoch, verbose = False):

        total_loss = torch.tensor(0., device = self.device)
        total_count = 0 # torch.tensor(0., device = self.device, dtype = torch.long)
        
        tracker = RateTracker() 
        if train: 
            self.model.train()

        for step_i, dat in enumerate(self.train_device_loader if train else self.test_device_loader): 

            # we don't actually need lengths so could stop returning? 
            species, tissue, assay, one_hot = dat

            if train: 
                self.optimizer.zero_grad()

            one_hot = one_hot.permute(0,2,1) # this is somewhat dumb: it's undone immediately in the model
            output = self.model(one_hot) # B x C x T

            seq_out_norm = output - output.logsumexp(1, keepdims = True)
            one_hot_chomped = one_hot[:,:,1:]
            loss = - (one_hot_chomped * seq_out_norm[:,:,:-1]).sum() / (one_hot_chomped.sum() + 1e-8) 

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
        if self.xla: 
            print(f"device {xm.get_ordinal()}:", end = " ")
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

    if True: 
        flags = { 
            "use_xla" : True,
            "batch_size" : 100, 
            "data_parallel" : False
        } 
    else:
        flags = { 
            "use_xla" : False,
            "batch_size" : 10, 
            "data_parallel" : False
        } 

    print(flags)
    
    if flags["data_parallel"]: 
        xmp.spawn(_mp_fn, args=(flags,))
    else: 
        _mp_fn(0, flags)
