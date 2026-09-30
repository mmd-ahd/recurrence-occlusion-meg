# Third-party code

| Path | Origin | Notes |
|------|--------|-------|
| `dfbmodels/` | Deep Feedback Models, Calhas & Oliveira (2025) | Feedback-state and equilibrium modules and the ResNet with a feedback loop, adapted here: the feedback signal is added to the ResNet input instead of concatenated, and the feedback state is a 256-D bottleneck. Check the upstream licence before redistributing. |
| `cornet/` | CORnet, Kubilius et al. (2019), DiCarlo lab | Unmodified model definitions (MIT licence upstream). Pretrained weights are downloaded from the CORnet model zoo. |
| `BLT-VS-main/` | BLT-VS, KietzmannLab | Vendored copy with its own `LICENSE` (MIT) and README. Weights are downloaded from the Hugging Face hub (`novelmartis/blt_vs_model`). |

References:
- Calhas D., Oliveira A. L. (2025). Deep Feedback Models.
- Kubilius J. et al. (2019). Brain-like object recognition with high-performing shallow recurrent ANNs. NeurIPS.
