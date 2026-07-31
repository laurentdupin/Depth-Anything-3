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
outside this lean inference slice.

The additive ABI 2 `da3_create_vulkan` path maps the same canonical
safetensors file directly into a dependency-free Vulkan full graph. It keeps
all intermediate transformer and DPT tensors on the selected GPU; the current
tensor ABI performs only the caller's input upload and final depth download.
It fails rather than silently falling back to CPU.

| GPU | 28x28 relative L1 | 56x56 relative L1 |
|---|---:|---:|
| Radeon RX 9070 | `0.006247%` | `0.021963%` |
| GeForce GTX 1080 | `0.006253%` | `0.021954%` |
| Radeon RX 6700 XT | `0.006225%` | `0.021952%` |

Twenty consecutive 56x56 calls on persistent contexts completed on every
device. The concurrent canary medians were 23.87 ms (RX 9070), 29.91 ms
(GTX 1080), and 22.14 ms (RX 6700 XT); these are stability canaries rather
than isolated performance benchmarks.

The FP32 GPU baseline covers camera-token replacement, Q/K normalization,
local and global RoPE, local/global capture concatenation, the complete main
DPT branch, UV embeddings, and exponential metric depth. Mixed precision
remains disabled until a separate accuracy gate is added.

The additive InferBridge GPU-resource path imports a producer-owned shared
D3D12 BGRA texture and fence directly into Vulkan, performs preprocessing and
the complete graph without host tensor staging, and exports a leased shared
D3D12 R32_FLOAT texture plus completion fence. The executor records operators
and inter-operator buffer copies as separate command buffers and chains them by
GPU timeline semaphore values; it performs no CPU wait between operators and
does not call `vkQueueWaitIdle`.

The RX 9070 public-boundary canary retains three simultaneous output leases,
rejects a fourth job, verifies stable slot reuse, correlation, cancellation,
and lease survival after model/runtime shutdown. Across its repeated frames,
the process-wide tensor upload and download counters both remain unchanged.
The maximum normalized-range deviation from the scalar CPU result is
`0.00180328` (`0.180328%`). Full evidence is recorded in
`native/validation/inferbridge_gpu_canary_rx9070_2026-07-31.json`.

## InferBridge image contract

ABI 3 adds `da3_inferbridge_image_shape` and
`da3_infer_bgra8_f32`. The latter matches the current Python worker rather
than assuming an idealized RGB input: it preserves the first three BGR bytes
from the BGRA capture as the RGB-ordered numpy image consumed by DA3, applies
upper-bound resizing, rounds each dimension to a multiple of 14, and performs
ImageNet normalization. Depth remains at the processed resolution, matching
the worker's output header.

For a deterministic 83x61 BGRA image at process resolution 56, Python CPU and
Vulkan on the RX 9070, GTX 1080, and RX 6700 XT all produced 42x56 depth. The
image entry point also applies the worker's final per-image min/max
normalization, so its returned FP32 payload is directly compatible with the
InferBridge worker contract. Across the three devices, maximum absolute
deviation was `0.001280`, mean absolute deviation was at most `0.000679`, and
mean relative deviation was at most `0.1781%`. Relative error close to the
zero endpoint is ill-conditioned; the normalized-range maximum absolute
deviation is `0.1280%`.
`native/tools/validate_image_path.py` reproduces this canary.

## Embedded InferBridge harness

The model DLL itself exports `ibrh_get_api` for InferBridge harness ABI 1.0.
The implementation accepts one host-memory BGRA8 image and returns one leased
host-memory normalized FP32 depth image while preserving `source_frame_id` and
timestamp correlation. Runtime/device selection, canonical safetensors model
loading, submit/poll/acquire/release lifetimes, and stable error reporting are
covered by the harness tests. The output lease retains its backing job after
the caller releases the job handle.

Windows Vulkan builds advertise the common D3D12 GPU-resource capability in
addition to the host path. Runtime creation requires exact D3D12/Vulkan LUID
matching and fails if the selected adapter cannot import both the shared input
texture/fence and shared R32 output texture/fence. GPU submission never falls
back to host memory. Three reusable output slots are exposed; each lease owns
only its slot until release.

With the pinned canonical DA3-Small checkpoint, the Windows Release build
passes `da3_c_abi_smoke`, `da3_harness_abi_smoke`,
`da3_harness_full_graph`, and the hardware-gated
`da3_d3d12_full_graph`. The latter uses the exact common InferBridge API and
validates the complete shared-texture path on an RX 9070.
