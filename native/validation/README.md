# Depth Anything 3 Small native validation

The correctness-first target is the Apache-2.0 `DA3-SMALL` model at Hugging
Face revision `e08cab65ca0ec38e7826075418411ab90cab4da3`.

## Canonical model boundary

- Configuration SHA-256:
  `a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2`
- Configuration bytes: 1,202
- Canonical `model.safetensors` SHA-256:
  `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`
- Canonical model bytes: 137,248,940
- Tensors: 437 contiguous FP32 tensors

Unlike canonical PyTorch `.pth` files, safetensors has a bounded,
non-executable data format. The dependency-free native reader therefore maps
the canonical file directly: no conversion, hidden duplicate, pickle parser,
or Python deployment code is required. It validates the bounded JSON header,
F32 dtype, ranks, dimensions, integer arithmetic, unique names, contiguous
non-overlapping offsets, exact tensor byte counts, and full payload coverage.

The real catalog checkpoint passes the native model probe.

## Single-view full-graph gate

The dependency-free scalar oracle implements the complete DA3-Small backbone
used by InferBridge's single-image channel: ViT-S/14 patch and position
embedding, camera-token substitution, twelve transformer blocks, Q/K
normalization, local/global 2D RoPE semantics, alternating attention, and
the concatenated local/global captures at blocks 5, 7, 9, and 11. It also
implements the main DualDPT branch: per-level projection and resizing,
UV positional embedding, four refinement stages, and exponential metric-depth
output.

The exported DLL inference entry point was compared with PyTorch CPU using
deterministic normalized FP32 tensors:

| Input | Relative L1 | Maximum absolute error |
|---:|---:|---:|
| 28x28 | `2.67512e-7` (`0.0000268%`) | `4.17233e-7` |
| 56x56 | `6.18421e-8` (`0.00000618%`) | `3.57628e-7` |

Camera, ray, Gaussian-splat, confidence, and multi-view outputs are not
required by InferBridge's single-image depth channel and are intentionally
outside this lean inference slice. The present implementation is a
correctness-first CPU oracle. It does not advertise Vulkan or zero-copy GPU
capability.
