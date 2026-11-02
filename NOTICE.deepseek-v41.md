# DeepSeek V4.1 native training sources

The native model architecture and the test-only PyTorch oracle are adapted from
DeepSeek-V4.1-Flash `inference/model.py` and `inference/kernel.py`, revision
`314113f8a7dadb388215396ba5fbee2c610a04d6` (MIT).
The original license is preserved in `licenses/DeepSeek-V4.1-LICENSE`.

The training bindings call these separately installed upstream libraries:

| Library | Inspected revision | Use |
| --- | --- | --- |
| deepseek-ai/TileKernels | c1cf914a82e6901c679445f872051ce44ac2cc63 | mHC/Sinkhorn, RMSNorm, SwiGLU, RoPE, router forward/backward |
| deepseek-ai/DeepGEMM | 5081f8513af57e61696de4f4f04b0e707ffb503e | BF16 grouped forward, input gradients, weight gradients |
| NVIDIA/Megatron-LM | 2fcef8359b977750f1868dbc5766baa9c26b2fb0 | fused token permutation and packed CSA attention |
| NVIDIA/TransformerEngine | 63451d6b868af0db6dd879cf5dd6a3f9bf14e1f9 | Megatron's fused permutation and wide RMSNorm |
| sgl-project/sglang | a4e51e2873174a98021bd79640bc826d69a10941 | native V4.1 inference and layer-input capture reference |

No DeepSpec operator source is vendored. No `tile_kernels.torch` reference
implementation is imported in the production path. The fork retains the
upstream SpecForge package/CLI name for compatibility.
