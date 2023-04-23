import os
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
import torch.optim as optim

# running: OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 distributed.py

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def main():
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    is_distributed = num_gpus > 1
    if is_distributed:
        torch.cuda.set_device(args.local_rank)  # 这里设定每一个进程使用的GPU是一定的
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    model = ToyModel()
    model = model.cuda()
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=True)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    while True:
        optimizer.zero_grad()
        outputs = model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(args.local_rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()


if __name__ == '__main__':
    main()