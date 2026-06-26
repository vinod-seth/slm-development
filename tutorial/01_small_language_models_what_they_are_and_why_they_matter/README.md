# Module 1: Small Language Models: What They Are and Why They Matter

Establish a clear mental model of small language models, where they fit in the AI ecosystem, and how this course is structured for beginners.

## Lessons

1. [SLMs vs LLMs: Scope, Trade-offs, and Real Use Cases](01_slms_vs_llms_scope_trade_offs_and_real_use_cases.md)
   - *~25 minutes*
2. [How a Language Model Learns: Tokens, Probabilities, and Prediction](02_how_a_language_model_learns_tokens_probabilities_and_prediction.md)
   - *~30 minutes*
3. [Setting Up Your Training Environment](03_setting_up_your_training_environment.md)
   - *~25 minutes*
4. [Module Quiz](quiz.md)

## Module Objectives

By the end of this module you will be able to:
- Distinguish small language models from large language models using parameter count, hardware requirements, and deployment context
- Identify three categories of real-world tasks where <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLMs</abbr> outperform <abbr title="Large Language Model: a massive language model (7B+ parameters) requiring cloud or cluster hardware.">LLMs</abbr> on latency, privacy, or domain-scope grounds
- Name at least four SLM model families covered in this course (SmolLM2, Phi-2, TinyLlama, DistilGPT-2) and their approximate sizes
- Explain <abbr title="The preprocessing step of converting raw text input into numerical tokens that a language model can process.">tokenization</abbr> and describe what a token is without using jargon
- Describe the next-token prediction objective and why it drives language model training
- Interpret a simple probability distribution over a vocabulary as a model output
- Recognize how byte-pair encoding (BPE) compresses vocabulary — citing Sennrich et al. (2016)
- Create an isolated Python virtual environment and confirm the correct PyTorch and Transformers versions are installed
- Verify <abbr title="Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning.">GPU</abbr> availability (or configure <abbr title="Central Processing Unit: the general-purpose processor in a computer.">CPU</abbr> fallback) using a five-line diagnostic script
- Navigate the course repository directory structure and locate each module folder
- Use the provided devcontainer or Dockerfile for zero-friction cloud setup

---

[← Back to Course](../../README.md)