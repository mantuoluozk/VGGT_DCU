import importlib.metadata
import platform
from pathlib import Path

import flash_attn
import torch
import torchvision

from common import command, parser, save, setup, sha256

args = parser("Phase 1: capture DCU environment and artifact identities").parse_args()
setup(args)
data = {
    "status": "PASS", "python": platform.python_version(), "torch": torch.__version__,
    "platform": platform.platform(), "torch_build_config": torch.__config__.show(),
    "module_paths": {"torch": torch.__file__, "torchvision": torchvision.__file__, "flash_attn": flash_attn.__file__},
    "torchvision": torchvision.__version__, "hip": torch.version.hip,
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
    "source_commit": command("git", "-C", str(args.source), "rev-parse", "HEAD"),
    "source_status": command("git", "-C", str(args.source), "status", "--porcelain"),
    "checkpoint": {"path": str(args.checkpoint), "bytes": args.checkpoint.stat().st_size, "sha256": sha256(args.checkpoint)},
    "devices": [{"index": i, "name": torch.cuda.get_device_name(i),
                 "memory_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],
    "backend_flags": {"flash": torch.backends.cuda.flash_sdp_enabled(), "math": torch.backends.cuda.math_sdp_enabled()},
    "test_script_sha256": {path.name: sha256(path) for path in sorted(Path(__file__).resolve().parent.glob("*.py"))},
}
assert len(data["devices"]) == 8, "Expected 8 DCUs for this acceptance run"
assert data["source_commit"]["code"] == 0 and data["source_status"]["code"] == 0
save(args, "environment", data)
