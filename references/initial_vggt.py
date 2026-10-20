"""Real-weight VGGT Flash SDPA verification, without customer compatibility patches."""
import gc
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vggt"))
from vggt.models.vggt import VGGT

OUT = ROOT / "retest_results"
OUT.mkdir(exist_ok=True)
result = {"torch": torch.__version__, "dtype": "bf16", "checkpoint": str(ROOT / "vggt_ckpt/model.pt"), "runs": []}
torch.manual_seed(42)
model = VGGT()
state = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
del state
model = model.cuda().eval()
print("Real weights loaded strictly", flush=True)
aggregator_backend = None
original_aggregator_forward = model.aggregator.forward

def aggregator_forward(*args, **kwargs):
    if aggregator_backend is None:
        return original_aggregator_forward(*args, **kwargs)
    with sdpa_kernel(aggregator_backend):
        return original_aggregator_forward(*args, **kwargs)

model.aggregator.forward = aggregator_forward
result["backend_scope"] = "Force aggregator only; original FP32 camera head may use MATH"

@torch.inference_mode()
def run(images, backend):
    global aggregator_backend
    aggregator_backend = backend
    allowed = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH] if backend == SDPBackend.FLASH_ATTENTION else [SDPBackend.MATH]
    try:
        with sdpa_kernel(allowed), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(images)
    finally:
        aggregator_backend = None

# Same inputs and weights; this is backend agreement, not task accuracy validation.
images = torch.rand(2, 3, 224, 224, device="cuda")
reference = {k: v.float().cpu() for k, v in run(images, SDPBackend.MATH).items() if isinstance(v, torch.Tensor)}
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    actual = run(images, SDPBackend.FLASH_ATTENTION)
    torch.cuda.synchronize()
result["forced_flash_ops"] = [x.key for x in prof.key_averages() if "attention" in x.key.lower()]
result["backend_agreement"] = {}
for key in ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"):
    a, b = actual[key].float().cpu(), reference[key]
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), key
    result["backend_agreement"][key] = {
        "shape": list(a.shape), "max_abs": (a-b).abs().max().item(),
        "relative_l2": ((a-b).norm()/b.norm().clamp_min(1e-20)).item(),
    }
print(json.dumps(result["backend_agreement"], indent=2), flush=True)
del actual, reference, images
gc.collect()
torch.cuda.empty_cache()

# Verify automatic dispatch independently of the forced-backend run.
images = torch.rand(1, 3, 224, 224, device="cuda")
with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        automatic = model(images)
    torch.cuda.synchronize()
result["automatic_ops"] = [x.key for x in prof.key_averages() if "attention" in x.key.lower()]
assert any("flash_attention" in op for op in result["automatic_ops"])
del automatic, images

for frames in (1, 4, 8, 16):
    images = torch.rand(frames, 3, 518, 518, device="cuda")
    try:
        for _ in range(2):
            run(images, SDPBackend.FLASH_ATTENTION)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(3):
            prediction = run(images, SDPBackend.FLASH_ATTENTION)
            del prediction
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 3
        row = {"frames": frames, "resolution": 518, "ms": elapsed*1000, "fps": frames/elapsed,
               "peak_allocated_GB": torch.cuda.max_memory_allocated()/1e9,
               "peak_reserved_GB": torch.cuda.max_memory_reserved()/1e9}
    except torch.OutOfMemoryError:
        row = {"frames": frames, "resolution": 518, "oom": True}
    result["runs"].append(row)
    print(json.dumps(row), flush=True)
    del images
    torch.cuda.empty_cache()
    (OUT / "vggt_flash_verification.json").write_text(json.dumps(result, indent=2))
    if row.get("oom"):
        break
print("VGGT FLASH VERIFICATION COMPLETE", flush=True)
