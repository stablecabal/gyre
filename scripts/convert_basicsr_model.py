import sys
from pathlib import Path

import safetensors.torch
import torch


def convert_model(path: Path, outdir: Path):
    t = torch.load(path)
    if "params_ema" in t:
        t = t["params_ema"]
    elif "params" in t:
        t = t["params"]
    elif "state_dict" in t:
        t = t["state_dict"]

    safetensors.torch.save_file(t, str(outdir / (path.stem + ".safetensors")))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: convert_basicsr_model.py [model] outdir")
        sys.exit(-1)

    args = [*sys.argv]
    _ = args.pop(0)

    outdir = Path(args.pop())
    if outdir.exists() and not outdir.is_dir():
        print(f"{outdir} exists and is not a directory")
        sys.exit(-1)

    if not outdir.exists():
        outdir.mkdir(parents=True)

    while args:
        arg = Path(args.pop())
        if arg.suffix in {".pt", ".pth", ".bin", ".ckpt"}:
            convert_model(Path(arg), Path(outdir))
        else:
            print(f"Don't know how to handle file with extension {arg.suffix}.")
