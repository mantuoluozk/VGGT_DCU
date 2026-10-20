"""Diagnostic: isolate RCCL communication and a tiny DDP model from VGGT."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist

torch.set_num_threads(8)
rank = int(os.environ["RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=timedelta(minutes=2))
sentinel = torch.tensor(float(rank + 1), device="cuda")
dist.all_reduce(sentinel)
assert sentinel.item() == 36
model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(64, 64).cuda(), device_ids=[rank])
model(torch.randn(2, 64, device="cuda")).square().mean().backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters())
print(f"rank {rank}: all_reduce=36 and tiny DDP backward PASS", flush=True)
dist.barrier()
dist.destroy_process_group()
