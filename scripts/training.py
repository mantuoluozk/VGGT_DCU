"""Phases 5-6: real MultitaskLoss, optimizer updates and DDP synchronization checks."""
import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from common import backend_scope, build, output_stats, parser, save, setup

p = parser("VGGT single-card / DDP training graph validation")
p.add_argument("--ddp", action="store_true")
p.add_argument("--steps", type=int, default=3)
p.add_argument("--frames", type=int, default=6)
p.add_argument("--initial-scale", type=float, default=65536.0)
p.add_argument("--anomaly", action="store_true")
p.add_argument("--init-threads", type=int, default=1)
p.add_argument("--contiguous-qkv", action=argparse.BooleanOptionalAction, default=True,
               help="Materialize Q/K/V before SDPA to avoid non-contiguous Flash backward NaNs")
p.add_argument("--backend", choices=["flash", "auto"], default="flash")
args = p.parse_args()
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
setup(args)
torch.autograd.set_detect_anomaly(args.anomaly)
if args.ddp:
    assert world == 8, "This acceptance test requires eight ranks"
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))

from loss import MultitaskLoss

print(f"rank {rank}: build model", flush=True)
# The image can crash in CPU trunc_normal_/erfinv with concurrent multi-threaded rank initialization.
torch.set_num_threads(args.init_threads)
model = build(args, training=True)
torch.set_num_threads(args.threads)
core = model
if args.ddp:
    print(f"rank {rank}: construct DDP", flush=True)
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False,
                                    gradient_as_bucket_view=True)
loss_fn = MultitaskLoss(camera={"weight": 5.0, "loss_type": "l1"},
                       depth={"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
                       point=None, track=None)
learning_rate, weight_decay = (1e-4, 0.01) if args.ddp else (5e-5, 0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=args.initial_scale)

# Distinct data per rank, unlike customer script. Parameter initialization remains synchronized by DDP.
torch.manual_seed(100 + rank)
B, S, H = 1, args.frames, 224
images = torch.rand(B, S, 3, H, H, device="cuda")
intrinsics = torch.eye(3, device="cuda").repeat(B, S, 1, 1)
intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = H * 0.8
intrinsics[..., 0, 2] = intrinsics[..., 1, 2] = H / 2
points = torch.rand(B, S, H, H, 3, device="cuda") * 4 - 2
batch = {"images": images, "intrinsics": intrinsics,
         "extrinsics": torch.eye(4, device="cuda").repeat(B, S, 1, 1),
         "depths": torch.rand(B, S, H, H, device="cuda") * 5 + 0.5,
         "world_points": points, "cam_points": points.clone(),
         "point_masks": torch.rand(B, S, H, H, device="cuda") > 0.1}
sentinel = torch.tensor(float(rank + 1), device="cuda")
if args.ddp:
    dist.all_reduce(sentinel)
assert sentinel.item() == world * (world + 1) / 2
data = {"status": "PASS", "rank": rank, "world_size": world, "frames": S, "resolution": H,
        "contiguous_qkv": args.contiguous_qkv, "initial_loss_scale": args.initial_scale,
        "optimizer": {"name": "AdamW", "lr": learning_rate, "weight_decay": weight_decay},
        "model_init_threads": args.init_threads, "compute_threads": args.threads,
        "backend": args.backend, "all_reduce_sentinel": sentinel.item(), "steps": [],
        "sync_scope": "257 evenly spaced elements from EVERY trainable parameter/gradient; exact all-rank comparison",
        "training_scope": "Random weights, synthetic GT, all aggregator parameters trainable; no full dataset training"}
parameters = [(name, parameter) for name, parameter in core.named_parameters() if parameter.requires_grad]


def signature(gradients=False):
    samples = []
    missing = []
    for name, parameter in parameters:
        value = parameter.grad if gradients else parameter
        if value is None:
            missing.append(name)
            continue
        flat = value.detach().reshape(-1)
        index = torch.linspace(0, flat.numel() - 1, min(257, flat.numel()), device="cuda").long()
        samples.append(flat[index].float())
    assert not missing, f"Missing gradients: {missing}"
    return torch.cat(samples)


def equality_error(sample):
    if not args.ddp:
        return 0.0
    low, high = sample.clone(), sample.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    return (high-low).abs().max().item()


for step in range(args.steps):
    optimizer.zero_grad(set_to_none=True)
    before = signature().clone()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with backend_scope(core, args.backend, contiguous_qkv=args.contiguous_qkv):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images=images)
            output_stats(predictions)
            losses = loss_fn(predictions, batch)
            loss = losses["objective"]
        assert torch.isfinite(loss), "Nonfinite loss"
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # All gradients checked for finiteness; sampling only concerns cross-rank equality.
    checks = [(name, torch.isfinite(parameter.grad).all()) for name, parameter in parameters if parameter.grad is not None]
    finite = torch.stack([flag for _, flag in checks]).tolist()
    bad = [name for (name, _), flag in zip(checks, finite) if not flag]
    if bad:
        save(args, f"training_failure_rank{rank}", {"status": "FAIL", "step": step, "bad_gradients": bad,
             "loss": loss.item(), "scale": scaler.get_scale(),
             "loss_components": {name: value.detach().item() for name, value in losses.items()}})
        raise AssertionError(f"Nonfinite gradients in {len(bad)} parameters; see failure JSON")
    gradient_sample = signature(gradients=True)
    assert gradient_sample.abs().max() > 0, "Zero gradients"
    gradient_error = equality_error(gradient_sample)
    assert gradient_error == 0, f"Gradient samples differ: {gradient_error}"
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    after = signature()
    assert torch.isfinite(after).all(), "Nonfinite parameter sample"
    update = (after-before).abs().max().item()
    assert update > 0 and scaler.get_scale() >= old_scale, "Optimizer skipped or did not update parameters"
    parameter_error = equality_error(after)
    assert parameter_error == 0, f"Parameter samples differ: {parameter_error}"
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    row = {"step": step, "loss": loss.item(), "seconds_including_checks": elapsed,
           "gradient_sample_max_rank_difference": gradient_error, "parameter_sample_max_rank_difference": parameter_error,
           "parameter_max_sample_update": update, "loss_scale": scaler.get_scale(),
           "loss_components": {name: value.detach().item() for name, value in losses.items()},
           "peak_allocated_GB": torch.cuda.max_memory_allocated()/1e9}
    if args.ddp:
        losses_all = [torch.zeros_like(loss) for _ in range(world)]
        dist.all_gather(losses_all, loss.detach())
        row["rank_losses"] = [value.item() for value in losses_all]
    data["steps"].append(row)
    save(args, f"ddp_rank{rank}" if args.ddp else "training_single", data)
    del predictions, losses, loss, before, after, gradient_sample

if args.ddp:
    dist.barrier()
    dist.destroy_process_group()
