"""Compare the InferBridge DA3 image path with the native Vulkan API."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--process-resolution", type=int, default=56)
    parser.add_argument("--width", type=int, default=83)
    parser.add_argument("--height", type=int, default=61)
    args = parser.parse_args()

    torchvision_library = torch.library.Library("torchvision", "DEF")
    torchvision_library.define(
        "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    try:
        import packaging  # noqa: F401
    except ModuleNotFoundError:
        import importlib
        import setuptools._vendor.packaging as packaging_vendor
        sys.modules["packaging"] = packaging_vendor
        for submodule in (
                "markers", "requirements", "specifiers", "tags",
                "utils", "version"):
            module = importlib.import_module(
                f"setuptools._vendor.packaging.{submodule}")
            setattr(packaging_vendor, submodule, module)
            sys.modules[f"packaging.{submodule}"] = module
    sys.path.insert(0, str((args.repo / "src").resolve()))
    from depth_anything_3.api import DepthAnything3

    rng = np.random.default_rng(20260730)
    bgra = rng.integers(
        0, 256, (args.height, args.width, 4), dtype=np.uint8)
    bgra[:, :, 3] = 255
    # This deliberately matches InferBridge: its first-three-byte BGR slice is
    # passed to DA3, whose numpy loader interprets it as RGB.
    python_image = bgra[:, :, :3]
    model = DepthAnything3.from_pretrained(
        str(args.checkpoint.resolve())).to("cpu")
    reference = model.inference(
        [python_image],
        process_res=args.process_resolution).depth[0].astype(np.float32)

    library = ctypes.CDLL(str(args.dll.resolve()))
    library.da3_create_vulkan.argtypes = [
        ctypes.c_char_p, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p)]
    library.da3_create_vulkan.restype = ctypes.c_int
    library.da3_infer_bgra8_f32.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64]
    library.da3_infer_bgra8_f32.restype = ctypes.c_int
    library.da3_last_error.restype = ctypes.c_char_p
    library.da3_destroy.argtypes = [ctypes.c_void_p]

    context = ctypes.c_void_p()
    status = library.da3_create_vulkan(
        str((args.checkpoint / "model.safetensors").resolve()).encode(),
        args.device, ctypes.byref(context))
    if status:
        raise RuntimeError(library.da3_last_error().decode())
    actual = np.empty_like(reference)
    try:
        status = library.da3_infer_bgra8_f32(
            context,
            bgra.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            bgra.strides[0], args.width, args.height,
            args.process_resolution,
            actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            actual.size)
        if status:
            raise RuntimeError(library.da3_last_error().decode())
    finally:
        library.da3_destroy(context)

    difference = np.abs(actual - reference)
    relative = difference / np.maximum(np.abs(reference), 1e-6)
    report = {
        "shape": list(reference.shape),
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "maximum_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
    }
    print(json.dumps(report, indent=2))
    if not np.isfinite(actual).all() or relative.mean() > 0.02:
        raise SystemExit("DA3 native image-path accuracy gate failed")


if __name__ == "__main__":
    main()
