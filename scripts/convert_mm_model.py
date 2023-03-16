import json
import os
import sys
from pathlib import Path

import safetensors.torch
import torch
from mmcv import Config


def convert_config(path: Path, outdir: Path):
    c = Config.fromfile(str(path))
    c.dump(str(outdir / (path.stem + ".yaml")))


def convert_model(path: Path, outdir: Path):
    t = torch.load(path)
    meta = {k: json.dumps(v) for k, v in t["meta"].items()}
    safetensors.torch.save_file(
        t["state_dict"], str(outdir / (path.stem + ".safetensors")), meta
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: convert_mm_model.py [config] [model] outdir")
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
        if arg.suffix in {".py", ".yaml"}:
            convert_config(arg, outdir)
        elif arg.suffix in {".pt", ".pth", ".bin", ".ckpt"}:
            convert_model(Path(arg), Path(outdir))
        else:
            print(f"Don't know how to handle file with extension {arg.suffix}.")
