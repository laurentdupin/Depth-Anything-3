"""Generate a deterministic DA3-Small single-view encoder fixture."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from depth_anything_3.cfg import create_object


def load_model(config_path: Path, weights_path: Path):
    config = json.loads(config_path.read_text())["config"]
    model = create_object(OmegaConf.create(config))
    state = load_file(weights_path)
    with weights_path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        metadata = json.loads(
            source.read(header_size)).get("__metadata__", {})
    for alias, target in metadata.items():
        state[alias] = state[target]
    state = {
        name.removeprefix("model."): value
        for name, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


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
        features, _ = model.backbone(value)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    value.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".input.bin"))
    feature_shapes = []
    feature_sums = []
    for index, (feature, _) in enumerate(features):
        feature.numpy().astype(np.float32).tofile(
            prefix.with_suffix(f".feature{index}.bin"))
        feature_shapes.append(list(feature.shape))
        feature_sums.append(float(feature.double().sum()))
    print(json.dumps({
        "input_shape": list(value.shape),
        "feature_shapes": feature_shapes,
        "feature_sums": feature_sums,
    }, indent=2))


if __name__ == "__main__":
    main()
