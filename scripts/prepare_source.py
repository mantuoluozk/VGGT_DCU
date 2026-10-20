"""Create a separate pinned VGGT checkout; never modify an existing source directory."""
import argparse
import subprocess
from pathlib import Path

COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
p = argparse.ArgumentParser()
p.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "vendor/vggt")
args = p.parse_args()
if args.destination.exists():
    raise SystemExit(f"Destination already exists: {args.destination}; use it via --source instead")
args.destination.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["git", "init", str(args.destination)], check=True)
subprocess.run(["git", "-C", str(args.destination), "remote", "add", "origin", "https://github.com/facebookresearch/vggt.git"], check=True)
subprocess.run(["git", "-C", str(args.destination), "fetch", "--depth", "1", "origin", COMMIT], check=True)
subprocess.run(["git", "-C", str(args.destination), "checkout", "--detach", "FETCH_HEAD"], check=True)
print(COMMIT)
