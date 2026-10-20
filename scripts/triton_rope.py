"""Inference-only 2D rotary embedding kernel with explicit FP16/BF16 rounding."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(X, POS, COS, SIN, OUT, TOTAL: tl.constexpr,
                 HEADS: tl.constexpr, LENGTH: tl.constexpr, DIM: tl.constexpr,
                 X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, X3: tl.constexpr,
                 P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                 BF16: tl.constexpr, FP32: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = i < TOTAL
    d = i % DIM
    n = (i // DIM) % LENGTH
    h = (i // (DIM*LENGTH)) % HEADS
    b = i // (DIM*LENGTH*HEADS)
    half: tl.constexpr = DIM//2
    quarter: tl.constexpr = DIM//4
    rotated_d = tl.where(d % half < quarter, d + quarter, d - quarter)
    axis = d // half
    position = tl.load(POS + b*P0 + n*P1 + axis*P2, valid, 0)
    cosine = tl.load(COS + position*half + d % half, valid, 0).to(tl.float32)
    sine = tl.load(SIN + position*half + d % half, valid, 0).to(tl.float32)
    x = tl.load(X + b*X0 + h*X1 + n*X2 + d*X3, valid, 0).to(tl.float32)
    rotated = tl.load(X + b*X0 + h*X1 + n*X2 + rotated_d*X3, valid, 0).to(tl.float32)
    rotated = tl.where(d % half < quarter, -rotated, rotated)
    first = x*cosine
    second = rotated*sine
    # Match the intermediate rounding of two separate native PyTorch multiplications.
    if FP32:
        pass
    elif BF16:
        first = first.to(tl.bfloat16).to(tl.float32)
        second = second.to(tl.bfloat16).to(tl.float32)
    else:
        first = first.to(tl.float16).to(tl.float32)
        second = second.to(tl.float16).to(tl.float32)
    tl.store(OUT + i, first + second, valid)


def fused_rope(tokens, positions, cosine, sine):
    assert tokens.ndim == 4 and tokens.shape[-1] % 4 == 0
    assert tokens.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert not torch.is_grad_enabled(), "This kernel only implements inference"
    output = torch.empty(tokens.shape, device=tokens.device, dtype=tokens.dtype)
    _rope_kernel[(triton.cdiv(output.numel(), 1024),)](
        tokens, positions, cosine, sine, output, output.numel(), tokens.shape[1], tokens.shape[2], tokens.shape[3],
        *tokens.stride(), *positions.stride(), tokens.dtype == torch.bfloat16, tokens.dtype == torch.float32, 1024,
        num_warps=4, enable_fp_fusion=False)
    return output


if __name__ == "__main__":
    from common import parser, save, setup
    args = parser("Validate fused 2D rotary embedding on DCU").parse_args()
    setup(args)
    from vggt.layers.rope import RotaryPositionEmbedding2D
    rope = RotaryPositionEmbedding2D()
    rows = []
    torch.manual_seed(42)
    with torch.inference_mode():
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for batch, length in ((1, 261), (8, 1374), (1, 43968)):
                for layout in ("contiguous", "transposed"):
                    x = torch.randn(batch, length, 16, 64, device="cuda", dtype=dtype).transpose(1, 2)
                    if layout == "contiguous":
                        x = x.contiguous()
                    positions = torch.randint(0, 38, (batch, length, 2), device="cuda")
                    cosine, sine = rope._compute_frequency_components(32, 38, x.device, dtype)
                    expected = rope(x, positions)
                    actual = fused_rope(x, positions, cosine, sine)
                    error = (actual.float()-expected.float()).abs().max().item()
                    rows.append({"dtype": str(dtype), "batch": batch, "length": length, "layout": layout, "max_abs": error})
                    assert torch.equal(expected, actual), rows[-1]
                    del x, positions, expected, actual
    save(args, "triton_rope_validation", {"status": "PASS", "tests": rows})
