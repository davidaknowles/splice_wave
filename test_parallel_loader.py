import torch
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def test_parallel_loader(rank):

    data = torch.arange(12).reshape(-1, 1)
    sampler = torch.utils.data.distributed.DistributedSampler(
        data,
        num_replicas=xr.world_size(),
        rank=rank,
        shuffle=True
    )
    
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler = sampler)

    parallel_loader = pl.MpDeviceLoader(dataloader, xm.xla_device())
    results = sum([batch[0].tolist() for batch in parallel_loader], [])

    print(f"Device {rank} received data: {results}")
    
    expected_data_size = len(data) // xr.world_size()
    print(f"Device {rank} received {len(results)} datapoints, expected {expected_data_size}")

if __name__ == "__main__":
    xmp.spawn(test_parallel_loader, args=()) 
