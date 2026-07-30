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

The real catalog checkpoint passes the native model probe. Graph inference
and numerical validation are pending, and no inference or GPU capability is
advertised by this foundation.
