import torchvision
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import denoising_diffusion_pytorch
import pandas as pd
import numpy as np
from torchvision import utils
import math
#from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.utils.data import TensorDataset, DataLoader

import importlib
importlib.reload(denoising_diffusion_pytorch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == "cpu":
    torch.set_num_threads(20)  

#train_ds = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()).data / 255.
test_ds = datasets.MNIST('../data', train=False, transform=transforms.ToTensor()).data / 255.

test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)

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

res_folder = trainer.results_folder / "impute"
res_folder.mkdir(exist_ok=True)

#num_samples = 1024

model = trainer.ema.ema_model

save_images = False

mae = {}

for U in [1, 3, 5, 10]: 

    errs = []
    for x_known in test_dl: 
        x_known = x_known[:,None,:,:]
        x_missing = x_known.clone()
        x_missing[:,:,:14,:] = 0.

        if save_images: 
            utils.save_image(x_known, res_folder / "original.png", nrow = int(math.sqrt(num_samples)))        
            utils.save_image(x_missing, res_folder / "missing.png", nrow = int(math.sqrt(num_samples)))
        
        x_missing[:,:,:14,:] = torch.nan
        
        x_sampled = model.p_conditional_sample_loop(x_missing.to(device), U = U).cpu()

        if save_images: 
            utils.save_image(x_sampled, res_folder / f"imputed{U}.png", nrow = int(math.sqrt(num_samples)))
        
        err = (x_sampled - x_known)[~x_missing.isnan()]

        errs.append( err.abs().mean().item() )

    mae[U] = np.mean( errs )
    
    df = pd.DataFrame(list(mae.items()), columns=['U', 'MAE'])
    
    # Save DataFrame to TSV file
    df.to_csv('impute_res.tsv', sep='\t', index=False)
