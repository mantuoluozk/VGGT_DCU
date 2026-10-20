"""Measure VGGT inference candidates against an unchanged real-weight baseline."""
import contextlib
import gc
import json
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from common import backend_scope, build, command, parser, save, setup, sha256

p = parser("Profile and compare output-preserving VGGT inference candidates")
p.add_argument("--frames", nargs="+", type=int, default=[8, 32])
p.add_argument("--resolution", type=int, default=518)
p.add_argument("--variants", nargs="+", default=["baseline", "contiguous", "direct_flash", "chunk16", "chunk32", "precast_linear"])
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--iterations", type=int, default=3)
p.add_argument("--profile", action="store_true")
p.add_argument("--real-images", action="store_true")
p.add_argument("--strict", action="store_true", help="Fail if any candidate errors or output comparison fails")
args = p.parse_args()
if not args.variants or args.variants[0] != "baseline":
    p.error("--variants must start with baseline")
if args.iterations < 1 or args.warmup < 0:
    p.error("iterations must be positive and warmup nonnegative")
setup(args)
torch.set_num_threads(1)
model = build(args, weights=True)
torch.set_num_threads(args.threads)
data = {"weights": "checkpoint", "resolution": args.resolution, "warmup": args.warmup,
        "iterations": args.iterations, "runs": [], "profiles": [],
        "torch": torch.__version__, "checkpoint_bytes": args.checkpoint.stat().st_size,
        "source_commit": command("git", "-C", str(args.source), "rev-parse", "HEAD"),
        "script_sha256": {path.name: sha256(path) for path in Path(__file__).resolve().parent.glob("*.py")},
        "scope": "Same model outputs, all heads retained, BF16 aggregator and original FP32 heads"}
inputs = {}
for frames in args.frames:
    torch.manual_seed(2026 + frames)
    inputs[frames] = torch.rand(frames, 3, args.resolution, args.resolution, device="cuda", dtype=torch.bfloat16)
if args.real_images:
    from vggt.utils.load_fn import load_and_preprocess_images
    files = sorted((args.source / "examples/kitchen/images").glob("*"))[:4]
    inputs["kitchen4"] = load_and_preprocess_images([str(f) for f in files]).cuda()


@contextlib.contextmanager
def candidate(name):
    if name.startswith("preset_"):
        from inference_presets import inference_preset
        with inference_preset(model, name.removeprefix("preset_")):
            @torch.inference_mode()
            def invoke(images):
                with backend_scope(model, "flash"), torch.autocast("cuda", dtype=torch.bfloat16):
                    return model(images)
            yield invoke
        return
    restores = []
    original_sdpa = F.scaled_dot_product_attention
    try:
        if name in ("cache_rope", "cache_all", "balanced", "fused_rope", "fused_balanced"):
            rope = model.aggregator.rope
            original_rope = rope.forward
            original_aggregator = model.aggregator.forward
            current = {}
            def aggregator(images):
                current["max_position"] = max(images.shape[-2]//model.aggregator.patch_size,
                                              images.shape[-1]//model.aggregator.patch_size) + 1
                return original_aggregator(images)
            def rope_forward(tokens, positions):
                feature_dim = tokens.size(-1)//2
                cosine, sine = rope._compute_frequency_components(feature_dim, current["max_position"], tokens.device, tokens.dtype)
                if name in ("fused_rope", "fused_balanced"):
                    from triton_rope import fused_rope
                    return fused_rope(tokens, positions, cosine, sine)
                vertical, horizontal = tokens.chunk(2, dim=-1)
                vertical = rope._apply_1d_rope(vertical, positions[..., 0], cosine, sine)
                horizontal = rope._apply_1d_rope(horizontal, positions[..., 1], cosine, sine)
                return torch.cat((vertical, horizontal), dim=-1)
            model.aggregator.forward = aggregator
            rope.forward = rope_forward
            restores.extend([lambda: setattr(rope, "forward", original_rope),
                             lambda: setattr(model.aggregator, "forward", original_aggregator)])
        if name in ("cache_position", "cache_all", "balanced", "fused_balanced"):
            from vggt.heads.utils import create_uv_grid, position_grid_to_embed
            cache = {}
            for head in (model.depth_head, model.point_head):
                original = head._apply_pos_embed
                def cached(x, W, H, ratio=0.1):
                    key = (x.shape[-2], x.shape[-1], x.shape[1], W, H, x.dtype, x.device, ratio)
                    if key not in cache:
                        position = create_uv_grid(x.shape[-1], x.shape[-2], aspect_ratio=W/H, dtype=x.dtype, device=x.device)
                        position = position_grid_to_embed(position, x.shape[1])*ratio
                        cache[key] = position.permute(2, 0, 1)[None]
                    return x + cache[key]
                head._apply_pos_embed = cached
                restores.append(lambda h=head, fn=original: setattr(h, "_apply_pos_embed", fn))
        elif name.startswith("chunk"):
            chunk = int(name[5:])
            for head in (model.depth_head, model.point_head):
                original = head.forward
                def wrap(fn):
                    def forward(*a, **kw):
                        kw["frames_chunk_size"] = chunk
                        return fn(*a, **kw)
                    return forward
                head.forward = wrap(original)
                restores.append(lambda h=head, fn=original: setattr(h, "forward", fn))
        elif name == "direct_flash":
            from flash_attn import flash_attn_func
            def direct(q, k, v, *a, **kw):
                if q.dtype not in (torch.float16, torch.bfloat16):
                    return original_sdpa(q, k, v, *a, **kw)
                assert not a and kw.get("attn_mask") is None and not kw.get("enable_gqa", False)
                output = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                         dropout_p=kw.get("dropout_p", 0.0),
                                         softmax_scale=kw.get("scale"), causal=kw.get("is_causal", False))
                return output.transpose(1, 2)
            F.scaled_dot_product_attention = direct
        elif name == "conv_tune":
            original = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = True
            restores.append(lambda value=original: setattr(torch.backends.cudnn, "benchmark", value))
        elif name == "heads_channels_last":
            for head in (model.depth_head, model.point_head):
                for parameter in head.parameters():
                    if parameter.ndim == 4:
                        original = parameter.detach().cpu().contiguous()
                        parameter.data = parameter.data.contiguous(memory_format=torch.channels_last)
                        restores.append(lambda param=parameter, cpu=original: setattr(param, "data", cpu.to("cuda")))
        elif name == "precast_linear":
            # Only operations already autocast to BF16. Normalization, tokens and FP32 heads are unchanged.
            for module in model.aggregator.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    for parameter in module.parameters(recurse=False):
                        if parameter.is_floating_point():
                            original = parameter.detach().cpu()
                            parameter.data = parameter.data.to(torch.bfloat16)
                            restores.append(lambda param=parameter, cpu=original: setattr(param, "data", cpu.to("cuda")))
        elif name not in ("baseline", "contiguous", "persistent_autocast", "cache_rope", "fused_rope"):
            raise ValueError(name)
        if name in ("balanced", "fused_balanced"):
            for head in (model.depth_head, model.point_head):
                original = head.forward
                def wrap(fn):
                    def forward(*a, **kw):
                        kw["frames_chunk_size"] = 16
                        return fn(*a, **kw)
                    return forward
                head.forward = wrap(original)
                restores.append(lambda h=head, fn=original: setattr(h, "forward", fn))
        if name == "persistent_autocast":
            with torch.no_grad(), backend_scope(model, "flash"), torch.autocast("cuda", dtype=torch.bfloat16):
                yield lambda images: model(images)
        else:
            @torch.inference_mode()
            def invoke(images):
                with backend_scope(model, "flash", contiguous_qkv=name == "contiguous"), torch.autocast("cuda", dtype=torch.bfloat16):
                    return model(images)
            yield invoke
    finally:
        for restore in reversed(restores):
            restore()
        F.scaled_dot_product_attention = original_sdpa


def flatten(predictions):
    out = {}
    for name, value in predictions.items():
        if name == "images":
            continue
        if isinstance(value, list):
            out.update({f"{name}.{i}": t.detach().float().cpu() for i, t in enumerate(value)})
        elif isinstance(value, torch.Tensor):
            out[name] = value.detach().float().cpu()
    return out


def compare(predictions, reference):
    actual = flatten(predictions)
    assert actual.keys() == reference.keys(), "Changed output keys"
    metrics = {}
    for key, a in actual.items():
        b = reference[key]
        assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(), key
        difference = a - b
        metrics[key] = {"max_abs": difference.abs().max().item(),
                        "relative_l2": (difference.norm()/b.norm().clamp_min(1e-20)).item(),
                        "allclose": torch.allclose(a, b, rtol=2e-3, atol=2e-3)}
    return {"status": "PASS" if all(row["allclose"] for row in metrics.values()) else "DIFF", "outputs": metrics}


def profile(invoke, images):
    records = []
    handles = []
    groups = [("image_encoder", model.aggregator.patch_embed), ("aggregator_total", model.aggregator),
              ("camera_head", model.camera_head), ("depth_head", model.depth_head), ("point_head", model.point_head)]
    for kind, blocks in (("frame", model.aggregator.frame_blocks), ("global", model.aggregator.global_blocks)):
        for block in blocks:
            groups.extend([(f"{kind}_block", block), (f"{kind}_attention", block.attn), (f"{kind}_mlp", block.mlp)])
    for group, module in groups:
        stack = []
        def before(mod, inp, label=group, pending=stack):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            pending.append(event)
        def after(mod, inp, output, label=group, pending=stack):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            records.append((label, pending.pop(), end))
        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    try:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        predictions = invoke(images)
        end.record()
        torch.cuda.synchronize()
        totals = {}
        for label, a, b in records:
            totals[label] = totals.get(label, 0.0) + a.elapsed_time(b)
        del predictions
        return {"total_ms": start.elapsed_time(end), "module_ms": totals,
                "note": "Nested categories overlap: attention/MLP are included in their block totals"}
    finally:
        for handle in handles:
            handle.remove()


references = {}
for variant in args.variants:
    try:
        with candidate(variant) as invoke:
            for key, images in inputs.items():
                for _ in range(args.warmup):
                    invoke(images)
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for iteration in range(args.iterations):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    prediction = invoke(images)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start)*1000)
                    if iteration < args.iterations - 1:
                        del prediction
                peak = torch.cuda.max_memory_allocated()/1e9
                if variant == "baseline":
                    references[key] = flatten(prediction)
                    agreement = {"status": "REFERENCE"}
                else:
                    agreement = compare(prediction, references[key])
                del prediction
                row = {"variant": variant, "input": key, "times_ms": times, "median_ms": statistics.median(times),
                       "mean_ms": statistics.mean(times), "peak_allocated_GB": peak, "agreement": agreement}
                data["runs"].append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "agreement"}), flush=True)
                print(f"agreement: {agreement['status']}", flush=True)
                if variant == "baseline" and args.profile and isinstance(key, int):
                    data["profiles"].append({"input": key, **profile(invoke, images)})
                save(args, "optimization_screen", data)
    except Exception:
        data["runs"].append({"variant": variant, "status": "ERROR", "error": traceback.format_exc()})
        save(args, "optimization_screen", data)
    gc.collect()
    torch.cuda.empty_cache()
save(args, "optimization_screen", data)
if args.strict and any(row.get("status") == "ERROR" or row.get("agreement", {}).get("status") == "DIFF" for row in data["runs"]):
    raise SystemExit(1)
