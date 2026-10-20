"""Download the official checkpoint. HF_ENDPOINT / HF_TOKEN are read by huggingface_hub."""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "checkpoints/model.pt")
args = p.parse_args()
if args.destination.exists():
    raise SystemExit(f"Destination already exists: {args.destination}; use it via --checkpoint instead")
args.destination.parent.mkdir(parents=True, exist_ok=True)
cached = hf_hub_download(repo_id="facebook/VGGT-1B", filename="model.pt")
shutil.copyfile(cached, args.destination)
print(args.destination.resolve())
