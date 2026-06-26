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
- Estimate <abbr title="Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning.">GPU</abbr> <abbr title="Video Random Access Memory: high-speed memory on a GPU used to store model weights and activations during run time.">VRAM</abbr> requirements for a given model size using the bytes-per-parameter rule
- Select an appropriate model family (sub-100M, 100M–500M, 500M–3B) given a hardware constraint
- Explain why gradient checkpointing and mixed-precision training reduce memory usage
- Identify which course models run on <abbr title="Central Processing Unit: the general-purpose processor in a computer.">CPU</abbr>-only, 8 GB VRAM, and 16 GB VRAM setups
- Load a <abbr title="A model trained on a massive general dataset to learn language patterns before fine-tuning.">pretrained</abbr> SmolLM2-135M checkpoint using AutoModelForCausalLM and AutoTokenizer
- Inspect model architecture, total parameter count, and layer names programmatically
- Run a basic <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr> call and decode the output tokens to readable text
- Read a Hugging Face model card and locate training data, license, and known limitations

---

[← Back to Course](../../README.md)