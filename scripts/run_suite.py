"""Run every customer VGGT phase sequentially, retaining commands, logs and exit codes."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, default=ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
p.add_argument("--phases", nargs="+", choices=["environment", "flash", "smoke", "correctness", "benchmark_flash", "training", "ddp"],
               default=["environment", "flash", "smoke", "correctness", "benchmark_flash", "training", "ddp"])
args = p.parse_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(VGGT_SOURCE=str(args.source.resolve()), VGGT_CHECKPOINT=str(args.checkpoint.resolve()),
           VGGT_OUTPUT=str(args.output), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONUNBUFFERED="1")
env.setdefault("NCCL_SOCKET_IFNAME", "lo")
env["PYTHONFAULTHANDLER"] = "1"
base = [sys.executable, "-u"]
jobs = {
    "environment": base + [str(ROOT / "scripts/environment.py")],
    "flash": base + [str(ROOT / "scripts/flash.py")],
    "smoke": base + [str(ROOT / "scripts/inference.py"), "smoke"],
    "correctness": base + [str(ROOT / "scripts/inference.py"), "correctness"],
    "benchmark_flash": base + [str(ROOT / "scripts/inference.py"), "benchmark", "--backend", "flash", "--random-init"],
    "training": base + [str(ROOT / "scripts/training.py"), "--steps", "1", "--frames", "6"],
    "ddp": base + ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8", str(ROOT / "scripts/training.py"), "--ddp", "--steps", "3", "--frames", "4"],
}
manifest_path = args.output / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"phases": {}}
for phase in args.phases:
    command = jobs[phase]
    start = datetime.now(timezone.utc)
    print(f"START {phase}: {command}", flush=True)
    with (args.output / f"{phase}.log").open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
            log.write("\nTimed out after 3600 seconds\n")
    manifest["phases"][phase] = {"command": command, "start_utc": start.isoformat(),
                                "end_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code,
                                "status": "PASS" if code == 0 else "FAIL"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"END {phase}: {'PASS' if code == 0 else 'FAIL'}", flush=True)
subprocess.run([sys.executable, str(ROOT / "scripts/make_report.py"), str(args.output)], check=True)
raise SystemExit(0 if all(row["exit_code"] == 0 for row in manifest["phases"].values()) else 1)
