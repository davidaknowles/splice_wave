import torchvision
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import denoising_diffusion_pytorch
import pandas as pd
from torchvision import utils
import math
#from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

import importlib
importlib.reload(denoising_diffusion_pytorch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == "cpu":
    torch.set_num_threads(20)  

#train_ds = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()).data / 255.
test_ds = datasets.MNIST('../data', train=False, transform=transforms.ToTensor()).data / 255.

model = denoising_diffusion_pytorch.Unet(
    dim = 64,
    channels = 1,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True,
    self_condition = False
)

diffusion = denoising_diffusion_pytorch.GaussianDiffusion(
    model,
    image_size = 28,
    sampling_timesteps = 250
).to(device)

trainer = denoising_diffusion_pytorch.Trainer(
    diffusion,
    test_ds[:,None,:,:], # add a dummy channel dimension
    train_batch_size = 64,
    results_folder = "mnist_diffusion_results",
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,            # whether to calculate fid during training
    num_workers = 4
)

# 100 is the last checkpoint
trainer.load(100) # this handles moving model to device

s = 20
num_samples = s * s

x_known = test_ds[:s,None,:,:]

x_known = x_known.repeat(s, 1, 1, 1)

x_missing = x_known.clone()

mae = {}
res_folder = trainer.results_folder / f"impute_test"
res_folder.mkdir(exist_ok=True)
utils.save_image(x_known, res_folder / "original.png", nrow = int(math.sqrt(num_samples)))

x_missing[:,:,:14,:] = 0.
utils.save_image(x_missing, res_folder / "missing.png", nrow = int(math.sqrt(num_samples)))

x_missing[:,:,:14,:] = torch.nan

model = trainer.ema.ema_model

for U in [1]: # , 3, 10]:     
    x_sampled = model.p_conditional_sample_loop(x_missing.to(device), U = U).cpu()
    
    utils.save_image(x_sampled, res_folder / f"imputed{U}_big.png", nrow = int(math.sqrt(num_samples)))
    
    err = (x_sampled - x_known)[~x_missing.isnan()]
    
    print(U, err.abs().mean().item())
    
