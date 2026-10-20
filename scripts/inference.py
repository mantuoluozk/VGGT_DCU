"""Customer phases 2-4, with explicit assertions and same-image backend controls."""
import gc
import atexit
from contextlib import ExitStack
import time

import torch

from common import backend_scope, build, output_stats, parser, save, setup

p = parser("VGGT inference validation")
p.add_argument("phase", choices=["smoke", "correctness", "benchmark"])
p.add_argument("--backend", choices=["flash", "auto"], default="flash")
p.add_argument("--random-init", action="store_true", help="Benchmark with random weights, matching customer script")
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--iterations", type=int, default=5)
p.add_argument("--max-frames", type=int, default=128)
p.add_argument("--preset", choices=["default", "balanced", "fused"], default="default",
               help="Optional output-preserving inference optimization")
args = p.parse_args()
setup(args)
model = build(args, weights=args.phase == "correctness" or (args.phase == "benchmark" and not args.random_init))
runtime = ExitStack()
atexit.register(runtime.close)
if args.preset != "default":
    from inference_presets import inference_preset
    runtime.enter_context(inference_preset(model, args.preset))
result_name = f"{args.phase}_{args.backend}" + (f"_{args.preset}" if args.preset != "default" else "")
data = {"phase": args.phase, "backend": args.backend, "preset": args.preset, "status": "PASS", "runs": []}


@torch.inference_mode()
def infer(images, dtype=torch.bfloat16):
    backend = "math" if dtype == torch.float32 else args.backend
    with backend_scope(model, backend), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        return model(images)


def checked(frames, resolution, dtype):
    images = torch.rand(frames, 3, resolution, resolution, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    predictions = infer(images, dtype)
    torch.cuda.synchronize()
    row = {"frames": frames, "resolution": resolution, "dtype": str(dtype),
           "ms": (time.perf_counter() - start) * 1000, "outputs": output_stats(predictions)}
    assert list(predictions["depth"].shape) == [1, frames, resolution, resolution, 1]
    assert list(predictions["world_points"].shape) == [1, frames, resolution, resolution, 3]
    data["runs"].append(row)
    return images, predictions


if args.phase == "smoke":
    # Flash does not support FP32. These two calls intentionally allow auto/MATH.
    requested = args.backend
    args.backend = "auto"
    checked(4, 518, torch.float32)
    checked(1, 224, torch.float32)
    args.backend = requested
    checked(4, 224, torch.bfloat16)
    data["fp32_policy"] = "auto (FP32 requires MATH); BF16 uses requested backend"

elif args.phase == "correctness":
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    images, predictions = checked(6, 518, torch.float32)
    # Customer runs correctness in FP32. Preserve that, and test BF16 Flash additionally.
    extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    rotation = extrinsics[0, :, :3, :3]
    orthogonal_error = (rotation @ rotation.transpose(-1, -2) - torch.eye(3, device="cuda")).abs().max().item()
    determinants = torch.det(rotation)
    assert torch.isfinite(extrinsics).all() and torch.isfinite(intrinsics).all()
    assert orthogonal_error < 1e-2 and (determinants - 1).abs().max() < 0.1
    assert (intrinsics[..., 0, 0] > 0).all() and (intrinsics[..., 1, 1] > 0).all()
    data["camera"] = {"orthogonal_max_abs": orthogonal_error, "det_min": determinants.min().item(), "det_max": determinants.max().item()}
    del images, predictions
    base = torch.rand(1, 3, 518, 518, device="cuda").repeat(6, 1, 1, 1)
    prediction = infer(base, torch.float32)
    extrinsics, _ = pose_encoding_to_extri_intri(prediction["pose_enc"], (518, 518))
    pose_std = extrinsics[0].reshape(6, -1).std(dim=0).mean().item()
    depth_std = prediction["depth"][0].std(dim=0).mean().item()
    assert pose_std < 1e-2 and depth_std < 1e-2
    data["identical_frames"] = {"pose_std": pose_std, "depth_std": depth_std}
    del base, prediction
    checked(4, 224, torch.float32)
    # Real image smoke and same-input FP32 / BF16 backend agreement (not task accuracy).
    from vggt.utils.load_fn import load_and_preprocess_images
    paths = sorted((args.source / "examples/kitchen/images").glob("*"))[:4]
    assert len(paths) == 4, "Expected official kitchen examples"
    real = load_and_preprocess_images([str(f) for f in paths]).cuda()
    fp32 = {k: v.float().cpu() for k, v in infer(real, torch.float32).items() if isinstance(v, torch.Tensor)}
    actual = infer(real)
    output_stats(actual)
    data["real_images"] = {"paths": [str(f.relative_to(args.source)) for f in paths], "shape": list(real.shape)}
    data["fp32_bf16_relative_l2"] = {}
    for key in ("pose_enc", "depth", "world_points"):
        a, b = actual[key].float().cpu(), fp32[key]
        data["fp32_bf16_relative_l2"][key] = ((a-b).norm()/b.norm().clamp_min(1e-20)).item()
    del fp32, actual
    reference = {k: v.float().cpu() for k, v in infer(real).items() if isinstance(v, torch.Tensor)}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        actual = infer(real)
        torch.cuda.synchronize()
    data["attention_ops"] = [event.key for event in prof.key_averages() if "attention" in event.key]
    if args.backend == "flash":
        assert any("flash_attention" in op for op in data["attention_ops"])
    data["repeatability_max_abs"] = {k: (actual[k].float().cpu()-v).abs().max().item() for k, v in reference.items()}
    data["scope"] = "Finiteness, pose legality, self consistency, real-image smoke; no dataset accuracy GT"

else:
    data.update(weights="random" if args.random_init else "checkpoint", warmup=args.warmup, iterations=args.iterations,
                memory_unit="decimal GB; allocated and reserved are allocator statistics")

    def benchmark(frames, resolution):
        images = torch.rand(frames, 3, resolution, resolution, device="cuda", dtype=torch.bfloat16)
        for _ in range(args.warmup):
            infer(images)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.iterations):
            infer(images)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start)/args.iterations
        return {"frames": frames, "resolution": resolution, "ms": elapsed*1000, "fps": frames/elapsed,
                "peak_allocated_GB": torch.cuda.max_memory_allocated()/1e9,
                "peak_reserved_GB": torch.cuda.max_memory_reserved()/1e9, "status": "PASS"}

    for resolution in (518, 224):
        for frames in (1, 4, 8, 16, 32, 64, 128):
            if frames > args.max_frames:
                continue
            gc.collect()
            torch.cuda.empty_cache()
            try:
                row = benchmark(frames, resolution)
            except torch.OutOfMemoryError as error:
                row = {"frames": frames, "resolution": resolution, "status": "OOM", "error": str(error)}
            data["runs"].append(row)
            save(args, result_name, data)
            if row["status"] == "OOM":
                gc.collect()
                torch.cuda.empty_cache()
                break

save(args, result_name, data)
