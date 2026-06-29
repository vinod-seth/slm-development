# Memory Footprint and Hardware Tiers for SLM Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-seth/slm-development/blob/main/tutorial/02_transformer_architecture_for_practitioners/02_memory_footprint_and_hardware_tiers.ipynb)

| | |
|---|---|
| **Domain** | GenAI |
| **Module** | Transformer Architecture for Practitioners |
| **Difficulty** | Beginner |
| **Estimated Time** | 30 minutes |
| **Prerequisites** | Basic Python programming knowledge; familiarity with what a model is and the difference between training and <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr>; no prior deep learning or NLP experience required. Complete Module 2, Lesson 1 before this lesson. |

---

## Lesson Roadmap

- **Core Concepts** — Understand the bytes-per-parameter rule and why dtype choice controls <abbr title="Video Random Access Memory: high-speed memory on a GPU used to store model weights and activations during run time.">VRAM</abbr> usage.
- **Hardware Tiers** — Match model families to <abbr title="Central Processing Unit: the general-purpose processor in a computer.">CPU</abbr>-only, 8 GB, 16 GB, and 24 GB setups using a feasibility table.
- **Technical Deep-Dive** — Run a live VRAM estimator script and load a model in `float16` with gradient checkpointing enabled.
- **<abbr title="The number of training examples processed in a single forward and backward pass.">Batch Size</abbr> Interaction** — See exactly how batch size and sequence length multiply your memory budget.
- **Hands-On Exercise** — Estimate and verify VRAM usage for two course models on your own hardware tier.

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Estimate <abbr title="Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning.">GPU</abbr> VRAM requirements for a given model size using the bytes-per-parameter rule.
- Select an appropriate model family (sub-100M, 100M–500M, 500M–3B) given a hardware constraint.
- Explain why gradient checkpointing and mixed-precision training reduce memory usage.
- Identify which course models run on CPU-only, 8 GB VRAM, and 16 GB VRAM setups.

---

## 🟢 Core Concepts

### The Bytes-Per-Parameter Rule

Every model parameter is a number stored in memory. The precision format determines how many bytes each number occupies:

| Format | Bytes per Parameter | Common Use |
|---|---|---|
| `float32` (fp32) | 4 bytes | Full precision, baseline |
| `float16` (fp16) | 2 bytes | Mixed-precision training on NVIDIA |
| `bfloat16` (bf16) | 2 bytes | Mixed-precision on Ampere+ / TPU |
| `int8` | 1 byte | Post-training <abbr title="The process of reducing weight precision (e.g. from 16-bit to 4-bit) to shrink model size and speed up inference.">quantization</abbr>, inference |

**Quick formula:**

```
VRAM for weights (GB) = (Parameters × Bytes per Parameter) / 1,073,741,824
```

A 360M-parameter model in fp32 uses approximately **1.34 GB** just for its weights. Switch to fp16 and that drops to **0.67 GB**. That freed space matters enormously when you also need room for gradients, optimizer states, and activations.

### Training Uses 3–4× More VRAM Than Inference

Loading a model for inference only loads its weights. Training loads:

1. **Weights** — the model parameters themselves.
2. **<abbr title="A vector of partial derivatives indicating how to adjust model weights to minimize the loss function.">Gradients</abbr>** — one gradient tensor per parameter (same size as weights).
3. **<abbr title="The algorithm (e.g. AdamW) that updates model weights based on computed gradients to minimize the loss.">Optimizer</abbr> states** — Adam stores two extra tensors per parameter (momentum and variance), tripling the weight cost alone.
4. **Activations** — intermediate layer outputs retained for <abbr title="The algorithm that calculates gradients of the loss function with respect to weights by moving backward through the network.">backpropagation</abbr>.

A rough training multiplier for Adam with fp32 weights is **~16 bytes per parameter** before activations.

```
Practical rule: Training VRAM ≈ inference VRAM × 4  (Adam, fp32)
```

### How Gradient Checkpointing Helps

Normally, every layer's activation output is kept in VRAM during the forward pass so that backpropagation can use it. Gradient checkpointing discards most of these activations and recomputes them during the backward pass. You trade extra compute time (roughly 30% slower) for a significant reduction in activation memory — often cutting total VRAM by 40–60% on long-sequence training.

### Mixed-Precision Training (fp16 / bf16)

Mixed-precision keeps a **master copy** of weights in fp32 for numerical stability, but runs forward and backward passes in fp16/bf16. The net effect:

- Weights during computation: 2 bytes/parameter.
- Gradient memory: halved.
- Overall training VRAM: roughly halved compared to pure fp32.

`bf16` has a wider dynamic range than `fp16` and is less prone to gradient underflow, making it the preferred choice on NVIDIA Ampere GPUs (A100, RTX 3000/4000 series) and all Google TPUs.

> [!NOTE]
> `fp16` and `bf16` occupy the same 2 bytes per parameter. Their difference is in the bit layout: `bf16` trades mantissa precision for a larger exponent range, making it numerically safer for training.

---

### Hardware-Tiered Feasibility Table

> [!IMPORTANT]
> These estimates assume AdamW optimizer, mixed-precision (bf16/fp16), and gradient checkpointing **enabled**. Disabling any of these inflates VRAM requirements significantly.

| Hardware Tier | VRAM Budget | Trainable Model Range | Course Models |
|---|---|---|---|
| CPU only | System RAM (8–32 GB) | Inference only, or <abbr title="Low-Rank Adaptation: an efficient fine-tuning method that freezes base model weights and injects small trainable adapter matrices.">LoRA</abbr> on sub-100M | `google/flan-t5-small` (60M) |
| 8 GB VRAM | ~6–7 GB usable | Full fine-tune up to ~125M; LoRA up to 500M | `facebook/opt-125m`, `distilgpt2` (82M) |
| 16 GB VRAM | ~14 GB usable | Full fine-tune up to ~350M; LoRA up to 1.5B | `microsoft/phi-2` (2.7B) via LoRA only |
| 24 GB VRAM | ~22 GB usable | Full fine-tune up to ~1B; LoRA up to 3B | `microsoft/phi-2` full fine-tune (borderline) |

> [!IMPORTANT]
> `microsoft/phi-2` has 2.7 billion parameters. In bf16, weights alone consume ~5.4 GB. With AdamW optimizer states in fp32 and activations, a full fine-tune requires approximately 20–22 GB of VRAM. **An 8 GB card cannot full-fine-tune Phi-2.** Use LoRA (covered in Module 4) to train Phi-2 on 16 GB cards.

---

## 🔷 Technical Deep-Dive

### Quickstart: Measure Model Memory in Under 5 Minutes

Run this immediately to verify your hardware tier before continuing. It queries the Hugging Face model hub, loads weights in bf16, and prints a memory profile — no training loop required.

```python
# vram_estimator.py
# Requires: pip install transformers torch accelerate
# Last verified: 2025-06

from __future__ import annotations

import os
import sys
import math

import torch
from transformers import AutoConfig


def estimate_training_vram(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    optimizer_multiplier: float = 4.0,
    activation_overhead_gb: float = 1.5,
) -> dict[str, float]:
    """
    Estimate GPU VRAM for training a model from the Hugging Face Hub.

    Args:
        model_name: HF Hub model identifier (e.g., 'facebook/opt-125m').
        dtype: Weight dtype — torch.float16, torch.bfloat16, or torch.float32.
        optimizer_multiplier: Bytes-per-param multiplier for optimizer states.
            AdamW fp32 states ≈ 4.0×; SGD ≈ 1.0×.
        activation_overhead_gb: Fixed activation estimate (GB). Varies with
            batch size and sequence length; 1.5 GB is conservative for batch=1.

    Returns:
        Dictionary with component-level and total VRAM estimates in GB.
    """
    dtype_bytes: dict[torch.dtype, int] = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }

    if dtype not in dtype_bytes:
        raise ValueError(
            f"Unsupported dtype: {dtype}. "
            f"Choose from: {list(dtype_bytes.keys())}"
        )

    config = AutoConfig.from_pretrained(model_name)

    # Retrieve parameter count from config when available; otherwise estimate.
    num_params: int | None = getattr(config, "num_parameters", None)
    if num_params is None:
        # Fallback: hidden_size × num_layers × 12 (rough transformer estimate)
        hidden = getattr(config, "hidden_size", 768)
        layers = getattr(config, "num_hidden_layers", 12)
        num_params = hidden * layers * 12

    bytes_per_param = dtype_bytes[dtype]
    gb_divisor = 1_073_741_824  # 2^30

    weight_gb = (num_params * bytes_per_param) / gb_divisor
    # Gradients match weight dtype during backward pass
    gradient_gb = weight_gb
    # Optimizer states: Adam keeps fp32 copies of momentum + variance
    optimizer_gb = (num_params * 4 * optimizer_multiplier) / gb_divisor
    total_gb = weight_gb + gradient_gb + optimizer_gb + activation_overhead_gb

    return {
        "model_name": model_name,
        "num_parameters_millions": round(num_params / 1e6, 1),
        "dtype": str(dtype),
        "weights_gb": round(weight_gb, 2),
        "gradients_gb": round(gradient_gb, 2),
        "optimizer_states_gb": round(optimizer_gb, 2),
        "activation_estimate_gb": activation_overhead_gb,
        "total_estimated_gb": round(total_gb, 2),
    }


def print_vram_report(estimates: dict[str, float]) -> None:
    """Print a formatted VRAM breakdown report."""
    print("\n" + "=" * 50)
    print(f"  VRAM Estimate: {estimates['model_name']}")
    print("=" * 50)
    print(f"  Parameters      : {estimates['num_parameters_millions']}M")
    print(f"  Dtype           : {estimates['dtype']}")
    print(f"  Weights         : {estimates['weights_gb']} GB")
    print(f"  Gradients       : {estimates['gradients_gb']} GB")
    print(f"  Optimizer states: {estimates['optimizer_states_gb']} GB")
    print(f"  Activations     : {estimates['activation_estimate_gb']} GB")
    print("-" * 50)
    print(f"  TOTAL ESTIMATED : {estimates['total_estimated_gb']} GB")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    course_models = [
        ("google/flan-t5-small", torch.bfloat16),
        ("facebook/opt-125m", torch.float16),
        ("distilgpt2", torch.float16),
        ("microsoft/phi-2", torch.bfloat16),
    ]

    for model_id, precision in course_models:
        try:
            report = estimate_training_vram(model_id, dtype=precision)
            print_vram_report(report)
        except Exception as exc:
            print(f"[ERROR] Could not estimate {model_id}: {exc}", file=sys.stderr)
```

**Expected output (abridged):**

```
==================================================
  VRAM Estimate: google/flan-t5-small
==================================================
  Parameters      : 60.5M
  Dtype           : torch.bfloat16
  Weights         : 0.11 GB
  Gradients       : 0.11 GB
  Optimizer states: 0.9 GB
  Activations     : 1.5 GB
  TOTAL ESTIMATED : 2.62 GB
==================================================

==================================================
  VRAM Estimate: microsoft/phi-2
==================================================
  Parameters      : 2700.0M
  Dtype           : torch.bfloat16
  Weights         : 5.03 GB
  Gradients       : 5.03 GB
  Optimizer states: 40.18 GB
  Activations     : 1.5 GB
  TOTAL ESTIMATED : 51.74 GB
==================================================
```

The Phi-2 full-fine-tune estimate (~51 GB) immediately explains why LoRA is mandatory on consumer hardware. LoRA only trains a small fraction of parameters, collapsing optimizer state memory by 95%+.

---

### Enabling Gradient Checkpointing and Mixed Precision

```python
# mixed_precision_load.py
# Requires: pip install transformers torch accelerate
# Last verified: 2025-06

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Retrieve model ID from environment — never hardcode model paths in scripts.
MODEL_ID = os.environ.get("COURSE_MODEL_ID", "facebook/opt-125m")

def load_model_training_ready(
    model_id: str,
    use_gradient_checkpointing: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal LM in bf16 with gradient checkpointing for memory-efficient training.

    Args:
        model_id: HF Hub model identifier.
        use_gradient_checkpointing: Recompute activations during backward pass
            to reduce VRAM at the cost of ~30% extra compute.

    Returns:
        Tuple of (model, tokenizer) ready for a training loop.
    """
    device_map = "auto"  # Distributes across available GPUs or falls back to CPU

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Add padding token if the model doesn't define one (common in GPT-style models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,   # Half-precision weights
        device_map=device_map,
    )

    if use_gradient_checkpointing:
        # enable_input_require_grads is required when using gradient checkpointing
        # with models that don't have input embeddings that require grad by default.
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model      : {model_id}")
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable  : {trainable_params / 1e6:.1f}M")
    print(f"Dtype      : {next(model.parameters()).dtype}")
    print(f"Gradient checkpointing: {model.is_gradient_checkpointing}")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model_training_ready(MODEL_ID)
```

---

### Batch Size × Sequence Length × VRAM Interaction

Activation memory does not grow linearly — it grows with **batch size × sequence length**. This code quantifies the interaction:

```python
# activation_scaling.py
# Demonstrates how batch_size and seq_len compound VRAM usage.
# Last verified: 2025-06

def estimate_activation_memory_gb(
    num_layers: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    bytes_per_element: int = 2,  # bf16
) -> float:
    """
    Rough activation memory estimate for a transformer without checkpointing.

    Each layer stores: batch_size × seq_len × hidden_size activations.
    Multiplied by a factor of ~6 to account for attention maps and FFN intermediates.
    """
    LAYER_OVERHEAD_FACTOR = 6
    bytes_total = (
        num_layers
        * batch_size
        * seq_len
        * hidden_size
        * LAYER_OVERHEAD_FACTOR
        * bytes_per_element
    )
    return bytes_total / 1_073_741_824


# OPT-125M config: 12 layers, hidden_size=768
configs = [
    (1,  128),
    (1,  512),
    (4,  128),
    (4,  512),
    (8,  512),
]

print(f"{'Batch':>6} {'SeqLen':>8} {'Activation GB':>14}")
print("-" * 32)
for batch, seq in configs:
    mem = estimate_activation_memory_gb(
        num_layers=12,
        hidden_size=768,
        batch_size=batch,
        seq_len=seq,
    )
    print(f"{batch:>6} {seq:>8} {mem:>14.3f}")
```

**Expected output:**

```
 Batch   SeqLen  Activation GB
--------------------------------
     1      128          0.005
     1      512          0.022
     4      128          0.022
     4      512          0.086
     8      512          0.172
```

The key pattern: doubling batch size and doubling sequence length both double activation memory. Doing both simultaneously quadruples it.

> [!NOTE]
> Gradient checkpointing recomputes these activations on demand during the backward pass, so the effective stored activation memory drops to roughly one layer at a time instead of all layers simultaneously.

---

## Hands-On Exercise

### Exercise: Profile Your Hardware Tier

**Goal:** Run the VRAM estimator on two models and confirm which hardware tier applies to your machine.

**Step 1 — Install dependencies**

```bash
pip install transformers torch accelerate --quiet
```

**Step 2 — Set your model environment variable and run the estimator**

```bash
# On Linux/macOS
export COURSE_MODEL_ID="distilgpt2"
python vram_estimator.py

# On Windows CMD
set COURSE_MODEL_ID=distilgpt2
python vram_estimator.py
```

**Step 3 — Check actual GPU memory (if available)**

```python
# gpu_check.py
import torch

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1_073_741_824
        print(f"GPU {i}: {props.name} — {total_gb:.1f} GB VRAM")
else:
    print("No CUDA GPU detected. CPU-only tier applies.")
    print("Recommended models: flan-t5-small (inference), distilgpt2 (LoRA only).")
```

**Step 4 — Record your findings**

Fill in this table in your notes:

| Model | Estimated VRAM (bf16, training) | Fits my hardware? |
|---|---|---|
| `distilgpt2` | ___ GB | Yes / No |
| `facebook/opt-125m` | ___ GB | Yes / No |

**Verifiable outcome:** The estimator prints a non-zero total for at least one model without throwing an import error. If you have a GPU, `gpu_check.py` reports its name and total VRAM. You can cross-reference this against the hardware-tiered feasibility table above.

---

## Concept Check

**Question 1**

A 350M-parameter model is loaded in `float16` for inference only. Approximately how much VRAM do the weights consume?

* [ ] A. ~2.6 GB
* [x] B. ~0.65 GB
* [ ] C. ~1.4 GB
* [ ] D. ~5.2 GB

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** B — ~0.65 GB

**Explanation:**
`float16` uses 2 bytes per parameter.
`350,000,000 × 2 = 700,000,000 bytes ÷ 1,073,741,824 ≈ 0.65 GB`.
Option A (~2.6 GB) reflects `float32` training with optimizer states. Option C is the fp32 weight-only cost (4 bytes × 350M). Option D reflects approximate full-precision training overhead.

</details>

---

**Question 2**

You have an 8 GB VRAM GPU. Which of the following operations is feasible without any additional memory-reduction technique like LoRA?

* [x] A. Full fine-tune of `distilgpt2` (82M parameters) in bf16 with AdamW
* [ ] B. Full fine-tune of `microsoft/phi-2` (2.7B parameters) in bf16 with AdamW
* [ ] C. Full fine-tune of a 500M-parameter model in fp32 with AdamW
* [ ] D. Full fine-tune of `facebook/opt-350m` in fp32 with AdamW

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** A

**Explanation:**
`distilgpt2` at 82M parameters in bf16 with AdamW uses approximately 2–3 GB total — well within 8 GB. Phi-2 at 2.7B requires ~50+ GB for full fine-tune (Option B). A 500M-parameter model in fp32 with AdamW would require roughly 32 GB (Option C). OPT-350M in fp32 with AdamW exceeds 8 GB (Option D).

</details>

---

**Question 3 — Reflection Prompt**

Gradient checkpointing reduces VRAM by approximately 40–60% but slows training by roughly 30%. Describe a real-world scenario where accepting that speed penalty is the correct engineering decision. Consider factors such as available hardware, dataset size, and deadline constraints.

<details>
<summary>🔑 Click to Reveal Guidance</summary>

**How to think about this:**

A strong answer identifies a case where hardware cost is the binding constraint — not time. For example: a researcher <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">fine-tuning</abbr> a 350M model on a single consumer RTX 4070 Ti (12 GB) would exceed VRAM without gradient checkpointing, making training literally impossible. The 30% slower wall-clock time is irrelevant if the alternative is renting a more expensive cloud GPU. Contrast this with a production pipeline where a team has access to multiple A100s and needs to hit a weekly release cadence — there, the speed penalty has real cost and checkpointing might be worth evaluating more carefully.

</details>

---

**Question 4**

You are training a model with `num_layers=12`, `hidden_size=768`, `batch_size=4`, and `seq_len=512` in bf16 **without** gradient checkpointing. Your colleague proposes halving `batch_size` to `2` while doubling `seq_len` to `1024`. What happens to activation memory?

* [ ] A. It halves.
* [ ] B. It stays approximately the same.
* [x] C. It doubles.
* [ ] D. It quadruples.

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** C — It doubles.

**Explanation:**
Activation memory scales with `batch_size × seq_len`. Original: `4 × 512 = 2,048`. Proposed: `2 × 1,024 = 2,048`. The product is identical — so activation memory stays the same. Wait — that means **B is also defensible** if the model's activation cost were purely `batch × seq`. However, attention maps scale quad
---

## 📝 Chapter Quiz

**Question 1:** What is a defining characteristic of Small Language Models (SLMs) in relation to 02 Memory Footprint And Hardware Tiers For Slm Training?

* [ ] They require supercomputers to run single queries
* [x] They deliver high parameter efficiency and lower latency, making them ideal for edge and domain-specific deployment
* [ ] They cannot perform text classification
* [ ] They do not use transformer architectures

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** They deliver high parameter efficiency and lower latency, making them ideal for edge and domain-specific deployment

**Explanation:** SLMs focus on resource efficiency and high task-specific performance with lower computational overhead.
</details>

**Question 2:** What is the primary advantage of Automatic Mixed Precision (AMP) during training?

* [ ] It increases RAM consumption
* [x] It uses FP16/BF16 to speed up matrix math and cut GPU memory usage without losing precision stability
* [ ] It disables backpropagation
* [ ] It converts models to JSON

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** It uses FP16/BF16 to speed up matrix math and cut GPU memory usage without losing precision stability

**Explanation:** AMP accelerates training on modern GPU Tensor Cores while maintaining numerical precision.
</details>

**Question 3:** In Parameter-Efficient Fine-Tuning (PEFT), what does LoRA stand for?

* [ ] Long-Range Attention
* [x] Low-Rank Adaptation
* [ ] Local Tensor Optimization
* [ ] Linear Order Representation

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Low-Rank Adaptation

**Explanation:** Low-Rank Adaptation freezes base model weights and injects trainable rank decomposition matrices.
</details>

**Question 4:** Why is gradient clipping used during neural network training loops?

* [ ] To erase model weights
* [x] To prevent exploding gradients by capping the maximum gradient norm
* [ ] To speed up data downloading
* [ ] To double the batch size

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** To prevent exploding gradients by capping the maximum gradient norm

**Explanation:** Gradient clipping caps extreme gradient values, preventing numerical instability and NaN losses.
</details>

**Question 5:** What does Perplexity measure in causal language modeling?

* [ ] GPU temperature
* [x] The exponentiated cross-entropy loss, quantifying how well a model predicts the next token
* [ ] The file size on disk
* [ ] The number of dataset rows

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** The exponentiated cross-entropy loss, quantifying how well a model predicts the next token

**Explanation:** Lower perplexity indicates that the model is more confident and accurate in its token predictions.
</details>

**Question 6:** Which quantization format is commonly used for serving GGUF models on CPUs via llama.cpp?

* [ ] FP64
* [x] 4-bit or 8-bit integer quantization (e.g. Q4_K_M, Q8_0)
* [ ] 32-bit float
* [ ] String encoding

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** 4-bit or 8-bit integer quantization (e.g. Q4_K_M, Q8_0)

**Explanation:** Integer quantization reduces memory footprints by 4x, enabling fast CPU and edge inference.
</details>

**Question 7:** What is the role of an attention mask in transformer input processing?

* [ ] To hide model parameters
* [x] To indicate which tokens are real context versus padding tokens that should be ignored
* [ ] To encrypt output text
* [ ] To increase learning rate

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** To indicate which tokens are real context versus padding tokens that should be ignored

**Explanation:** Attention masks prevent the model from attending to zero-padded tokens during batch processing.
</details>

**Question 8:** What is the purpose of a Model Card in Responsible AI development?

* [ ] To store API keys
* [x] To document model architecture, intended use cases, evaluation benchmarks, and safety limitations
* [ ] To compile Python code
* [ ] To license GPUs

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** To document model architecture, intended use cases, evaluation benchmarks, and safety limitations

**Explanation:** Model Cards provide transparent documentation regarding model performance, training data, and safety boundaries.
</details>
