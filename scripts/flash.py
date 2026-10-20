"""Flash-only forward/backward validation: forced SDPA vs direct flash_attn_func."""
import json
import os
from pathlib import Path

import flash_attn
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.set_num_threads(8)
torch.manual_seed(42)
output = Path(os.getenv("VGGT_OUTPUT", Path(__file__).resolve().parents[1] / "results/latest"))
output.mkdir(parents=True, exist_ok=True)
data = {"torch": torch.__version__, "flash_attn": flash_attn.__version__, "status": "PASS", "tests": []}
for dtype in (torch.float16, torch.bfloat16):
    q, k, v = [torch.randn(1, 16, 512, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        actual = F.scaled_dot_product_attention(q, k, v)
    upstream = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, (q, k, v), upstream)
    direct_inputs = [t.detach().transpose(1, 2).contiguous().requires_grad_() for t in (q, k, v)]
    direct = flash_attn_func(*direct_inputs, dropout_p=0.0, causal=False)
    direct_gradients = torch.autograd.grad(direct, direct_inputs, upstream.transpose(1, 2).contiguous())
    assert torch.isfinite(actual).all() and torch.isfinite(direct).all()
    assert all(torch.isfinite(g).all() for g in (*gradients, *direct_gradients))
    forward_error = ((actual.float()-direct.transpose(1, 2).float()).norm()/actual.float().norm()).item()
    gradient_errors = [((a.float()-b.transpose(1, 2).float()).norm()/a.float().norm()).item() for a, b in zip(gradients, direct_gradients)]
    assert forward_error < 0.02 and max(gradient_errors) < 0.02
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
    ops = [event.key for event in prof.key_averages() if "attention" in event.key]
    assert any("flash_attention" in op for op in ops)
    data["tests"].append({"dtype": str(dtype), "status": "PASS", "forward_relative_l2": forward_error,
                          "gradient_relative_l2": gradient_errors, "ops": ops})
(output / "flash_verification.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
print(json.dumps(data, indent=2), flush=True)
