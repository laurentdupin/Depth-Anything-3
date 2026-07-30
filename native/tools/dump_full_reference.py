"""Generate deterministic DA3-Small single-view depth fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_encoder_reference import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--size", type=int, default=28)
    args = parser.parse_args()

    model = load_model(args.config, args.weights)
    generator = torch.Generator().manual_seed(20260730)
    value = torch.randn(
        1, 1, 3, args.size, args.size, generator=generator)
    with torch.inference_mode():
        output = model(value)
    depth = output.depth
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    value.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".input.bin"))
    depth.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".depth.bin"))
    print(json.dumps({
        "input_shape": list(value.shape),
        "depth_shape": list(depth.shape),
        "minimum": float(depth.min()),
        "maximum": float(depth.max()),
        "mean": float(depth.mean()),
        "sum": float(depth.double().sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
