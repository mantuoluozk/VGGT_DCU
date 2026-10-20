"""Shared configuration, backend control and evidence output."""
import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--source", type=Path, default=Path(os.getenv("VGGT_SOURCE", ROOT / "vendor/vggt")))
    p.add_argument("--checkpoint", type=Path, default=Path(os.getenv("VGGT_CHECKPOINT", ROOT / "checkpoints/model.pt")))
    p.add_argument("--output", type=Path, default=Path(os.getenv("VGGT_OUTPUT", ROOT / "results/latest")))
    p.add_argument("--threads", type=int, default=8)
    return p


def setup(args):
    args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source))
    sys.path.insert(0, str(args.source / "training"))
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "DCU runtime unavailable"


def save(args, name, data):
    data = {"utc": datetime.now(timezone.utc).isoformat(), **data}
    target = args.output / f"{name}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"file": str(target), **data}, ensure_ascii=False), flush=True)


def command(*cmd):
    r = subprocess.run(cmd, text=True, capture_output=True)
    return {"code": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args, training=False, weights=False):
    from vggt.models.vggt import VGGT
    model = VGGT(enable_point=not training, enable_track=not training)
    if weights:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.cuda().train(training)


@contextlib.contextmanager
def backend_scope(model, backend):
    """Force only aggregator Flash: original FP32 camera head still needs MATH."""
    if backend == "auto":
        yield
        return
    if backend == "math":
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    originals = []

    def wrap(original):
        def forward(*args, **kwargs):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return original(*args, **kwargs)
        return forward

    # Layer wrappers also cover activation-checkpoint recomputation during backward.
    for module in model.aggregator.modules():
        if hasattr(module, "qkv") and hasattr(module, "fused_attn"):
            originals.append((module, module.forward))
            module.forward = wrap(module.forward)
    assert originals, "No aggregator attention modules found"
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            yield
    finally:
        for module, original in originals:
            module.forward = original


def output_stats(predictions):
    result = {}
    for key, value in predictions.items():
        tensors = value if isinstance(value, list) else [value]
        for index, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                continue
            assert torch.isfinite(tensor).all(), f"Nonfinite {key}"
            label = f"{key}[{index}]" if isinstance(value, list) else key
            result[label] = {"shape": list(tensor.shape), "min": tensor.min().item(), "max": tensor.max().item()}
    return result
