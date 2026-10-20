"""Small Flash-only shape probe retained to diagnose training backend failures."""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.set_num_threads(8)
rows = []
for batch, length in ((1, 261), (6, 261), (1, 1566), (1, 1044), (1, 512)):
    for gradient_scale in (1.0, 1e-6):
        q, k, v = [torch.randn(batch, 16, length, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = F.scaled_dot_product_attention(q, k, v)
        gradients = torch.autograd.grad(output, (q, k, v), torch.randn_like(output)*gradient_scale)
        rows.append({"batch": batch, "length": length, "gradient_scale": gradient_scale,
                     "output_finite": torch.isfinite(output).all().item(),
                     "gradients_finite": [torch.isfinite(g).all().item() for g in gradients]})
target = Path(os.getenv("VGGT_OUTPUT", ".")) / "flash_shape_diagnostic.json"
target.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
