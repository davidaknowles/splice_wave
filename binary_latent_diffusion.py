# Based on https://github.com/lucidrains/denoising-diffusion-pytorch
# Plus bits pulled from https://github.com/ZeWang95/BinaryLatentDiffusion (which is pretty complex overall) 
# Binary Latent Diffusion paper: https://arxiv.org/abs/2304.04820
# This implementation is just the diffusion model (DM) part so isn't actually latent

import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

from denoising_diffusion_1d import *

# from https://github.com/ZeWang95/BinaryLatentDiffusion/blob/1ae570226ce5147bdc90bbe92d3a19e4789a89e7/models/binarylatent.py#L232
class noise_scheduler(nn.Module):
    def __init__(self, steps=40, beta_type='linear', verbose = False):
        super().__init__()


        if beta_type == 'linear':

            beta = 1 - 1 / (steps - torch.arange(1, steps+1) + 1) 

            k_final = [1.0]
            b_final = [0.0]

            for i in range(steps):
                k_final.append(k_final[-1]*beta[i])
                b_final.append(beta[i] * b_final[-1] + 0.5 * (1-beta[i]))

            k_final = k_final[1:]
            b_final = b_final[1:]


        elif beta_type == 'cosine':

            k_final = torch.linspace(0.0, 1.0, steps+1)

            k_final = k_final * torch.pi
            k_final = 0.5 + 0.5 * torch.cos(k_final)
            b_final = (1 - k_final) * 0.5

            beta = []
            for i in range(steps):
                b = k_final[i+1] / k_final[i]
                beta.append(b)
            beta = torch.tensor(beta)

            k_final = k_final[1:]
            b_final = b_final[1:]
        
        elif beta_type == 'sigmoid':
            
            def sigmoid(x):
                z = 1/(1 + torch.exp(-x))
                return z

            def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=0.0):
                # A gamma function based on sigmoid function.
                v_start = sigmoid(start / tau)
                v_end = sigmoid(end / tau)
                output = sigmoid((t * (end - start) + start) / tau)
                output = (v_end - output) / (v_end - v_start)
                return torch.clip(output, clip_min, 1.) # was np
            
            k_final = torch.linspace(0.0, 1.0, steps+1)
            k_final = sigmoid_schedule(k_final, 0, 3, 0.8)
            b_final = (1 - k_final) * 0.5

            beta = []
            for i in range(steps):
                b = k_final[i+1] / k_final[i]
                beta.append(b)
            beta = torch.tensor(beta)

            k_final = k_final[1:]
            b_final = b_final[1:]

        else:
            raise NotImplementedError
        
        k_final = torch.hstack([torch.tensor(1.), k_final])
        b_final = torch.hstack([torch.tensor(0.), b_final])
        beta = torch.hstack([torch.tensor(1.), beta])
        self.register_buffer('k_final', k_final)
        self.register_buffer('b_final', b_final)
        self.register_buffer('beta', beta)  
        self.register_buffer('cumbeta', torch.cumprod(self.beta, 0))  
        # pdb.set_trace()

        if verbose: 
            print(f'Noise scheduler with {beta_type}:')
    
            print(f'Diffusion 1.0 -> 0.5:')
            data = (1.0 * self.k_final + self.b_final).data.numpy()
            print(' '.join([f'{d:0.4f}' for d in data]))
    
            print(f'Diffusion 0.0 -> 0.5:')
            data = (0.0 * self.k_final + self.b_final).data.numpy()
            print(' '.join([f'{d:0.4f}' for d in data]))
    
            print(f'Beta:')
            print(' '.join([f'{d:0.4f}' for d in self.beta.data.numpy()]))

    def one_step(self, x, t):
        dim = x.ndim - 1
        k = self.beta[t].view(-1, *([1]*dim))
        x = x * k + 0.5 * (1-k)
        return x

    def forward(self, x, t):
        dim = x.ndim - 1
        k = self.k_final[t].view(-1, *([1]*dim))
        b = self.b_final[t].view(-1, *([1]*dim))
        out = k * x + b
        return out

class BinaryLatentDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        loss_final = "mean", 
        objective = 'pred_noise',
        beta_schedule = 'cosine', # linear, cosine, sigmoid
        ddim_sampling_eta = 0.
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.num_timesteps = timesteps

        self.scheduler = noise_scheduler(self.num_timesteps, beta_type=beta_schedule)

        self.objective = objective

        self.loss_final = loss_final

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        self.p_flip = objective == 'pred_noise'

    @torch.no_grad()
    def sample(self, x, temp=1.0, sample_steps=None, return_all=False, device = 'cuda'):
        """ Would rather pass in a partial x here """

        b = x.shape[0] # batch size
        x_t = torch.bernoulli(0.5 * torch.ones_like(x))

        m = ~x.isnan()
        mask_flag = x.isnan().any()
        if mask_flag:
            #m = mask['mask'].unsqueeze(0)
            #latent = mask['latent'].unsqueeze(0)
            #x_t = x * m + x_t * (1-m)
            x_t[m] = x[m]
        
        sampling_steps = torch.arange(1, self.num_timesteps+1)
        sample_steps = default(sample_steps, self.num_timesteps)
        
        if sample_steps != self.num_timesteps:
            idx = torch.linspace(0.0, 1.0, sample_steps)
            idx = torch.tensor(idx * (self.num_timesteps-1), dtype=int)
            sampling_steps = sampling_steps[idx]

        if return_all:
            x_all = [x_t]

        sampling_steps = sampling_steps.flip(0)

        for i, t in tqdm(enumerate(sampling_steps)):
            t = torch.full((b,), t, device=device, dtype=torch.long)

            x_0_logits = self.model(x_t, t-1)
            x_0_logits = x_0_logits / temp # scale by temperature

            x_0_logits = torch.sigmoid(x_0_logits)

            if self.p_flip:
                x_0_logits =  x_t * (1 - x_0_logits) + (1 - x_t) * x_0_logits

            if not t[0].item() == 1:
                t_p = torch.full((b,), sampling_steps[i+1], device=device, dtype=torch.long)
                
                x_0_logits = torch.cat([x_0_logits.unsqueeze(-1), (1-x_0_logits).unsqueeze(-1)], dim=-1)
                x_t_logits = torch.cat([x_t.unsqueeze(-1), (1-x_t).unsqueeze(-1)], dim=-1)

                p_EV_qxtmin_x0 = self.scheduler(x_0_logits, t_p)
                q_one_step = x_t_logits

                for mns in range(sampling_steps[i] - sampling_steps[i+1]):
                    q_one_step = self.scheduler.one_step(q_one_step, t - mns)

                unnormed_probs = p_EV_qxtmin_x0 * q_one_step
                unnormed_probs = unnormed_probs / unnormed_probs.sum(-1, keepdims=True)
                unnormed_probs = unnormed_probs[...,0]
                
                x_tm1_logits = unnormed_probs
                x_tm1_p = torch.bernoulli(x_tm1_logits)
            
            else:
                x_0_logits = x_0_logits
                x_tm1_p = (x_0_logits > 0.5) * 1.0

            x_t = x_tm1_p

            if mask_flag:
                #x_t = latent * m + x_t * (1-m)
                x_t[m] = x[m]
                
            if return_all:
                x_all.append(x_t)
        if return_all:
            return torch.cat(x_all, 0)
        else:
            return x_t
    
    def forward(self, x_0):
        """ Label could be used for e.g. cell type """

        # x_0 = x_0.float() # convert bool 
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
        
        # make x noisy and denoise
        x_t = self.scheduler(x_0, t) # self.q_sample(x_0, t)
        
        x_t_in = torch.bernoulli(x_t) # add noise
        
        x_0_hat_logits = self.model(x_t_in, t-1)

        if self.p_flip: # this is like predicting noise instead of x_0
            x_0_hat_logits = x_t_in * ( - x_0_hat_logits) + (1 - x_t_in) * x_0_hat_logits
        
        kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')
        
        assert(kl_loss.max().isfinite())

        if self.loss_final == 'weighted':
            weight = (1 - ((t-1) / self.num_timesteps)).view(-1, 1, 1)
        elif self.loss_final == 'mean':
            weight = 1.0
        else:
            raise NotImplementedError
        
        loss = (weight * kl_loss).mean()
        kl_loss = kl_loss.mean()

        with torch.no_grad():
            acc = (((x_0_hat_logits > 0.0) * 1.0 == x_0) * 1.0).sum() / float(x_0.numel())

        return {'loss': loss, 'bce_loss': kl_loss, 'acc': acc}