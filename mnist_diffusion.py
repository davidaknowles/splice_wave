import torchvision
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import denoising_diffusion_pytorch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

import importlib
importlib.reload(denoising_diffusion_pytorch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_ds = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()).data / 255.
test_ds = datasets.MNIST('../data', train=False, transform=transforms.ToTensor()).data / 255.

model = Unet(
    dim = 64,
    channels = 1,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True,
    self_condition = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 28,
    sampling_timesteps = 250
).to(device)

trainer = Trainer(
    diffusion,
    train_ds[:,None,:,:], # add a dummy channel dimension
    train_batch_size = 64,
    results_folder = "mnist_diffusion_self_cond",
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,            # whether to calculate fid during training
    num_workers = 4
)

trainer.train()

