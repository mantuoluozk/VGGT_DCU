"""Verify container FlashAttention and forced PyTorch SDPA without compatibility shims."""
import json
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

OUT = Path(__file__).resolve().parent / "retest_results"
OUT.mkdir(exist_ok=True)
result = {
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "devices": [],
    "tests": [],
}
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    result["devices"].append({"index": i, "name": p.name, "memory_bytes": p.total_memory})
print(json.dumps(result, indent=2), flush=True)
torch.manual_seed(42)

for dtype in (torch.float16, torch.bfloat16):
    row = {"dtype": str(dtype)}
    try:
        q, k, v = [torch.randn(1, 16, 512, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(q, k, v)
        upstream = torch.randn_like(ref)
        ref_grads = torch.autograd.grad(ref, (q, k, v), upstream)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            flash = F.scaled_dot_product_attention(q, k, v)
        flash_grads = torch.autograd.grad(flash, (q, k, v), upstream)
        tolerance = 0.02 if dtype == torch.float16 else 0.08
        row["forward_max_abs"] = (flash.float() - ref.float()).abs().max().item()
        row["forward_relative_l2"] = ((flash.float() - ref.float()).norm() / ref.float().norm()).item()
        row["gradient_relative_l2"] = [((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20)).item() for a, b in zip(flash_grads, ref_grads)]
        assert torch.isfinite(flash).all() and all(torch.isfinite(g).all() for g in flash_grads)
        assert row["forward_relative_l2"] < tolerance
        assert max(row["gradient_relative_l2"]) < tolerance
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
        row["sdpa_ops"] = [x.key for x in prof.key_averages() if "attention" in x.key.lower()]
        row["forced_flash_sdpa"] = "PASS"
    except Exception:
        row["forced_flash_sdpa"] = "FAIL"
        row["sdpa_error"] = traceback.format_exc()
    try:
        import flash_attn
        from flash_attn import flash_attn_func
        row["flash_attn_version"] = flash_attn.__version__
        q2, k2, v2 = [t.detach().transpose(1, 2).contiguous().requires_grad_() for t in (q, k, v)]
        direct = flash_attn_func(q2, k2, v2, dropout_p=0.0, causal=False)
        direct.backward(upstream.transpose(1, 2).contiguous())
        torch.cuda.synchronize()
        assert torch.isfinite(direct).all() and all(torch.isfinite(t.grad).all() for t in (q2, k2, v2))
        row["direct_relative_l2"] = ((direct.transpose(1, 2).float() - ref.float()).norm() / ref.float().norm()).item()
        assert row["direct_relative_l2"] < tolerance
        row["direct_flash_attn"] = "PASS"
    except Exception:
        row["direct_flash_attn"] = "FAIL"
        row["direct_error"] = traceback.format_exc()
    result["tests"].append(row)
    print(json.dumps(row, indent=2), flush=True)
(OUT / "flash_verification.json").write_text(json.dumps(result, indent=2))
raise SystemExit(0 if len(result["devices"]) > 0 and all(t["forced_flash_sdpa"] == t["direct_flash_attn"] == "PASS" for t in result["tests"]) else 1)
