# Module 2: Transformer Architecture for Practitioners

Build enough architectural intuition about transformers to reason about model size, memory footprint, and training behavior without deriving equations from scratch.

## Lessons

1. [Inside a Transformer Block: Attention, FFN, and Residuals](01_inside_a_transformer_block_attention_ffn_and_residuals.md)
   - *~35 minutes*
2. [Memory Footprint and Hardware Tiers for SLM Training](02_memory_footprint_and_hardware_tiers_for_slm_training.md)
   - *~30 minutes*
3. [Loading and Inspecting Pretrained SLMs with Hugging Face](03_loading_and_inspecting_pretrained_slms_with_hugging_face.md)
   - *~30 minutes*
4. [Module Quiz](quiz.md)

## Module Objectives

By the end of this module you will be able to:
- Sketch the data flow through one transformer block: self-attention, feed-forward network, and residual connections
- Interpret the scaled dot-product attention formula at a conceptual level, with citation to Vaswani et al. (2017)
- Explain how layer count and hidden dimension jointly determine parameter count
- Differentiate encoder-only, decoder-only, and encoder-decoder architectures by use case
- Estimate GPU VRAM requirements for a given model size using the bytes-per-parameter rule
- Select an appropriate model family (sub-100M, 100M–500M, 500M–3B) given a hardware constraint
- Explain why gradient checkpointing and mixed-precision training reduce memory usage
- Identify which course models run on CPU-only, 8 GB VRAM, and 16 GB VRAM setups
- Load a pretrained SmolLM2-135M checkpoint using AutoModelForCausalLM and AutoTokenizer
- Inspect model architecture, total parameter count, and layer names programmatically
- Run a basic inference call and decode the output tokens to readable text
- Read a Hugging Face model card and locate training data, license, and known limitations

---

[← Back to Course](../../README.md)