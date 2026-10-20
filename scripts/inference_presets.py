"""Optional inference presets. Model weights, outputs and attention coverage are unchanged."""
import contextlib

import torch


@contextlib.contextmanager
def inference_preset(model, name="default", head_chunk_size=16):
    """Apply a reversible VGGT inference preset for the lifetime of this context.

    balanced: reuse position embeddings, avoid repeated GPU-to-host position reductions,
              process dense outputs in chunks of 16 frames.
    fused: balanced plus the inference-only Triton 2D rotary embedding kernel.
    """
    if name == "default":
        yield
        return
    if name not in ("balanced", "fused"):
        raise ValueError(name)
    assert not model.training, "Inference presets must not be applied to training models"
    if name == "fused":
        from triton_rope import fused_rope
    restores = []
    positions_cache = {}
    shape = {}
    rope = model.aggregator.rope
    original_aggregator = model.aggregator.forward
    original_rope = rope.forward

    def aggregator(images):
        shape["max_position"] = max(images.shape[-2]//model.aggregator.patch_size,
                                    images.shape[-1]//model.aggregator.patch_size) + 1
        return original_aggregator(images)

    def rotary(tokens, positions):
        if tokens.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return original_rope(tokens, positions)
        cosine, sine = rope._compute_frequency_components(tokens.shape[-1]//2, shape["max_position"],
                                                          tokens.device, tokens.dtype)
        if name == "fused":
            return fused_rope(tokens, positions, cosine, sine)
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = rope._apply_1d_rope(vertical, positions[..., 0], cosine, sine)
        horizontal = rope._apply_1d_rope(horizontal, positions[..., 1], cosine, sine)
        return torch.cat((vertical, horizontal), dim=-1)

    def cached_position(x, W, H, ratio=0.1):
        from vggt.heads.utils import create_uv_grid, position_grid_to_embed
        key = (x.shape[-2], x.shape[-1], x.shape[1], W, H, x.dtype, x.device, ratio)
        if key not in positions_cache:
            if len(positions_cache) >= 32:
                positions_cache.clear()
            position = create_uv_grid(x.shape[-1], x.shape[-2], aspect_ratio=W/H, dtype=x.dtype, device=x.device)
            position = position_grid_to_embed(position, x.shape[1])*ratio
            positions_cache[key] = position.permute(2, 0, 1)[None]
        return x + positions_cache[key]

    def chunked(original):
        def forward(*args, **kwargs):
            kwargs["frames_chunk_size"] = head_chunk_size
            return original(*args, **kwargs)
        return forward

    try:
        model.aggregator.forward = aggregator
        rope.forward = rotary
        restores.extend([(model.aggregator, "forward", original_aggregator), (rope, "forward", original_rope)])
        for head in (model.depth_head, model.point_head):
            if head is None:
                continue
            restores.extend([(head, "forward", head.forward), (head, "_apply_pos_embed", head._apply_pos_embed)])
            head.forward = chunked(head.forward)
            head._apply_pos_embed = cached_position
        yield
    finally:
        for module, attribute, original in reversed(restores):
            setattr(module, attribute, original)
        positions_cache.clear()
