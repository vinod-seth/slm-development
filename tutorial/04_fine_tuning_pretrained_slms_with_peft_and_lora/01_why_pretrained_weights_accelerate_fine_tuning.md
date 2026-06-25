# Why Pretrained Weights Accelerate Fine-Tuning

| | |
|---|---|
| **Domain** | GenAI |
| **Module** | Fine-Tuning Pretrained SLMs with PEFT and LoRA |
| **Difficulty** | Beginner |
| **Estimated Time** | 20 minutes |
| **Prerequisites** | Basic Python programming knowledge; familiarity with what a model is and the difference between training and inference; Module 3 completed with at least one successful training checkpoint saved |

---

## Lesson Roadmap

- **Section 1 — Core Concepts**: Understand why pretrained weights give you a head start and how transfer learning reduces your data burden.
- **Section 2 — Technical Deep-Dive**: Inspect a 135M-parameter SLM's memory footprint under full fine-tuning vs. LoRA, then load a pretrained checkpoint in Python.
- **Section 3 — Hands-On Exercise**: Run a script that prints frozen vs. trainable parameter counts for a SmolLM2-135M model.
- **Section 4 — Concept Check**: Verify your understanding with 4 questions.
- **Section 5 — Summary & References**: Consolidate takeaways and citations.

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain how pretrained weights encode transferable language knowledge and reduce data requirements for fine-tuning.
- Compare the GPU memory footprint of full fine-tuning vs. LoRA fine-tuning for a 135M-parameter model.
- Identify which transformer layers benefit most from task-specific adaptation.
- Confirm that your Module 3 training checkpoint exists before beginning any fine-tuning run.

---

> [!IMPORTANT]
> **Module 3 Gate**: This module assumes you have completed at least one full training run in Module 3 and have a saved checkpoint directory on disk. If you skipped Module 3, complete it before continuing — LoRA fine-tuning builds directly on those concepts and that saved state.

> [!IMPORTANT]
> **HF Hub Account**: From this module onward, some model weights require a Hugging Face account to access. Create one at [huggingface.co](https://huggingface.co) and run `huggingface-cli login` in your terminal before proceeding.

---

## 🟢 Core Concepts

### Transfer Learning: Borrowing What a Model Already Knows

Training a language model from scratch on domain-specific text is expensive. A 135M-parameter model trained on raw text requires billions of tokens and days of GPU time. Pretrained weights sidestep most of that cost.

Think of a pretrained model as a linguist who already reads fluently in English. They understand grammar, common phrasing, factual associations, and sentence structure. When you hire them for a specialist role — say, reading clinical trial reports — you don't re-teach English. You run a short orientation on domain vocabulary and task format. That orientation is fine-tuning.

This intuition maps directly to the mathematics. During pretraining, each transformer layer encodes a different level of linguistic abstraction:

```
Token Embeddings  →  Early Layers   →  Middle Layers  →  Late Layers
(vocabulary)         (syntax,            (semantics,        (task-specific
                      position)           coreference)       reasoning)
```

Early and middle layers learn broadly transferable representations. Late layers — particularly the final attention heads and the language model head — are where task-specific behavior lives. This is why PEFT methods such as LoRA (Hu et al., 2021) target a small subset of weight matrices rather than all parameters.

### How Pretrained Weights Reduce Your Data Requirement

Brown et al. (2020) demonstrated that pretrained models generalize to new tasks with far fewer examples than training from scratch requires. For a 135M-parameter SLM, the practical implication is concrete: a well-curated dataset of 5,000–20,000 task examples can produce a capable fine-tuned model when you start from pretrained weights. Starting from random initialization for the same task would require orders of magnitude more data.

> [!NOTE]
> This data-efficiency advantage is the primary reason SLMs are viable for edge deployment and private-data scenarios — you need less labeled data to reach acceptable task performance.

### Full Fine-Tuning vs. LoRA: A Memory Footprint Comparison

Full fine-tuning updates every parameter in the model. For a 135M-parameter model stored in `float32`, that is:

```
135,000,000 parameters × 4 bytes = ~540 MB (weights alone)
```

During training, the optimizer (e.g., AdamW) maintains two additional momentum states per parameter, multiplying the memory requirement:

```
Weights:           ~540 MB
Gradient buffer:   ~540 MB
AdamW states:     ~1,080 MB (2× weights)
─────────────────────────────
Total:            ~2,160 MB  (~2.1 GB) minimum
```

LoRA injects small trainable rank-decomposition matrices into targeted weight matrices and freezes everything else. With a typical LoRA rank of `r=8` applied to query and value projection matrices, the trainable parameter count drops to roughly 0.5–1% of total parameters — around 675K–1.35M for a 135M model. The optimizer states now apply only to those trainable parameters:

```
LoRA trainable params:   ~1.35M  → ~5.4 MB
Optimizer states:        ~10.8 MB
Frozen model (no grad):  ~540 MB (loaded, but no gradient storage)
─────────────────────────────
Total:                   ~556 MB  (~0.5 GB) practical minimum
```

This reduction makes fine-tuning feasible on a consumer GPU with 6–8 GB of VRAM.

> **Sidebar — VRAM Discussion (Optional Reading)**: VRAM requirements vary significantly based on batch size, sequence length, and mixed-precision settings (`bfloat16` vs `float32`). The figures above assume `float32` with batch size 1. In practice, using `bfloat16` halves weight storage to ~270 MB for a 135M model. See Module 5 for a complete quantization-aware training walkthrough.

---

## 🔷 Technical Deep-Dive

### Environment Check

Before running any code, verify your environment has the required packages. Run this once in your terminal:

```bash
# Verify package availability — expects no ImportError output
python -c "
import torch
import transformers
import peft
import datasets
print(f'torch       {torch.__version__}')
print(f'transformers {transformers.__version__}')
print(f'peft        {peft.__version__}')
print(f'datasets    {datasets.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

Expected output (versions may differ, CUDA optional for this lesson):

```
torch        2.3.x
transformers 4.44.x
peft         0.11.x
datasets     2.x.x
CUDA available: True   # or False on CPU-only machines
```

---

### Step 1 — Load a Pretrained SmolLM2-135M Checkpoint

`SmolLM2-135M` is a publicly accessible, non-gated model from Hugging Face (HuggingFaceTB/SmolLM2-135M, last verified 2025-07). It is an ideal teaching model: small enough to fit in CPU RAM, large enough to demonstrate real transfer learning behavior.

```python
# lesson4_1_pretrained_footprint.py
# Demonstrates pretrained weight loading and parameter audit for SmolLM2-135M.
# Requires: transformers>=4.44, peft>=0.11, torch>=2.1

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


MODEL_ID = "HuggingFaceTB/SmolLM2-135M"  # Non-gated; no HF login required
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# Target the query and value projection matrices — the most task-sensitive weights
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable_count, total_count) for a given model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def estimate_optimizer_memory_mb(trainable_params: int, dtype_bytes: int = 4) -> float:
    """
    Estimate AdamW optimizer memory for trainable parameters.
    AdamW stores 2 momentum states per parameter (first + second moment).
    """
    return (trainable_params * dtype_bytes * 2) / (1024 ** 2)


def main() -> None:
    print(f"Python {sys.version}")
    print(f"PyTorch {torch.__version__}\n")

    # --- Load base model (float32 for pedagogical clarity) ---
    print(f"Loading pretrained model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,   # explicit; avoid silent dtype assumptions
        low_cpu_mem_usage=True,       # stream weights to avoid peak RAM spike
    )

    trainable_base, total_base = count_parameters(base_model)
    weight_mb_base = (total_base * 4) / (1024 ** 2)
    optimizer_mb_base = estimate_optimizer_memory_mb(total_base)

    print("\n── Full Fine-Tuning Footprint ──────────────────────────")
    print(f"  Total parameters      : {total_base:,}")
    print(f"  Trainable parameters  : {trainable_base:,}  (all, in full FT)")
    print(f"  Weight storage        : {weight_mb_base:.1f} MB")
    print(f"  Gradient buffer       : {weight_mb_base:.1f} MB  (1× weights)")
    print(f"  AdamW optimizer states: {optimizer_mb_base:.1f} MB (2× weights)")
    print(f"  ─── Estimated total   : {weight_mb_base * 4:.1f} MB")

    # --- Apply LoRA configuration ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",   # do not train bias terms — keeps adapter weight count low
    )

    peft_model = get_peft_model(base_model, lora_config)

    trainable_lora, total_lora = count_parameters(peft_model)
    optimizer_mb_lora = estimate_optimizer_memory_mb(trainable_lora)
    trainable_pct = (trainable_lora / total_lora) * 100

    print("\n── LoRA Fine-Tuning Footprint ──────────────────────────")
    print(f"  Total parameters      : {total_lora:,}")
    print(f"  Trainable parameters  : {trainable_lora:,}  ({trainable_pct:.2f}% of total)")
    print(f"  Frozen weight storage : {weight_mb_base:.1f} MB (no gradients)")
    print(f"  LoRA weight storage   : {(trainable_lora * 4) / (1024**2):.1f} MB")
    print(f"  AdamW optimizer states: {optimizer_mb_lora:.1f} MB (LoRA params only)")
    print(f"  ─── Estimated total   : {weight_mb_base + (trainable_lora * 4) / (1024**2) + optimizer_mb_lora:.1f} MB")

    # --- Summarize which layers are frozen vs. trainable ---
    print("\n── Layer Trainability Summary (first 10 named modules) ──")
    for name, param in list(peft_model.named_parameters())[:10]:
        status = "TRAINABLE" if param.requires_grad else "frozen   "
        print(f"  [{status}]  {name}")

    # --- Quick forward pass to confirm the model loads correctly ---
    print("\n── Sanity Forward Pass ─────────────────────────────────")
    sample_text = "The quarterly compliance report highlights"
    inputs = tokenizer(sample_text, return_tensors="pt")
    with torch.no_grad():
        outputs = peft_model(**inputs)
    print(f"  Input tokens : {inputs['input_ids'].shape[1]}")
    print(f"  Logits shape : {outputs.logits.shape}  ✓")
    print("\nLesson 4-1 parameter audit complete.")


if __name__ == "__main__":
    main()
```

**Expected output (approximate — exact parameter counts depend on model revision):**

```
── Full Fine-Tuning Footprint ──────────────────────────
  Total parameters      : 134,515,456
  Trainable parameters  : 134,515,456  (all, in full FT)
  Weight storage        : 513.1 MB
  Gradient buffer       : 513.1 MB  (1× weights)
  AdamW optimizer states: 1,026.2 MB (2× weights)
  ─── Estimated total   : 2,052.4 MB

── LoRA Fine-Tuning Footprint ──────────────────────────
  Total parameters      : 134,909,952
  Trainable parameters  :     394,496  (0.29% of total)
  Frozen weight storage : 513.1 MB (no gradients)
  LoRA weight storage   :   1.5 MB
  AdamW optimizer states:   3.0 MB (LoRA params only)
  ─── Estimated total   : 517.6 MB

── Sanity Forward Pass ─────────────────────────────────
  Input tokens : 7
  Logits shape : torch.Size([1, 7, 49152])  ✓
```

> [!NOTE]
> The `total parameters` count increases slightly after `get_peft_model` because LoRA adds new adapter weight matrices. The frozen base weights do not accumulate gradients, so they do not contribute to optimizer memory.

---

### Step 2 — Verify Your Module 3 Checkpoint

Run this check before any fine-tuning run to confirm you have a valid starting point:

```python
# checkpoint_gate.py
# Gate script: confirms a Module 3 checkpoint exists before LoRA fine-tuning.
# Place this at the top of any Module 4 training script.

import sys
from pathlib import Path


def assert_checkpoint_ready(checkpoint_dir: str) -> Path:
    """
    Validate that a training checkpoint directory from Module 3 exists
    and contains the minimum expected files.

    Args:
        checkpoint_dir: Relative or absolute path to the checkpoint folder.

    Returns:
        Resolved Path object if valid.

    Raises:
        SystemExit: If the checkpoint is missing or incomplete.
    """
    checkpoint_path = Path(checkpoint_dir).resolve()
    required_files = {"config.json", "tokenizer.json"}

    if not checkpoint_path.exists():
        print(
            f"[ERROR] Checkpoint directory not found: {checkpoint_path}\n"
            "        Complete Module 3 and save a training checkpoint before continuing.",
            file=sys.stderr,
        )
        sys.exit(1)

    found_files = {f.name for f in checkpoint_path.iterdir()}
    missing = required_files - found_files

    if missing:
        print(
            f"[ERROR] Checkpoint directory is incomplete. Missing: {missing}\n"
            "        Re-run Module 3's training cell and confirm the checkpoint saves correctly.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[OK] Checkpoint verified at: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    # Replace with your actual Module 3 output path
    assert_checkpoint_ready("./module3_outputs/checkpoint-final")
```

---

## Hands-On Exercise

**Goal**: Confirm the memory footprint numbers from the deep-dive on your own machine and identify which specific weight matrices LoRA makes trainable.

**Time**: ~8 minutes

**Steps:**

1. **Clone the script** — Copy `lesson4_1_pretrained_footprint.py` into your working directory.

2. **Run the script**:
   ```bash
   python lesson4_1_pretrained_footprint.py
   ```

3. **Record your numbers** — In a text file or notebook cell, paste your actual output for:
   - Full fine-tuning estimated total MB
   - LoRA estimated total MB
   - Trainable parameter percentage

4. **Inspect the layer list** — Modify the script to print *all* named parameters (not just the first 10). Change line:
   ```python
   for name, param in list(peft_model.named_parameters())[:10]:
   ```
   to:
   ```python
   for name, param in peft_model.named_parameters():
   ```
   Re-run and count how many layers are marked `TRAINABLE`. Note which module names appear.

5. **Verify your checkpoint** — Run `checkpoint_gate.py` pointing to your Module 3 output directory. Confirm you see `[OK]`.

**Verifiable outcome**: Your terminal shows `Logits shape : torch.Size([1, N, 49152]) ✓` and `[OK] Checkpoint verified`. If either fails, resolve it before moving to Lesson 2.

> [!NOTE]
> The vocabulary size `49152` is specific to SmolLM2-135M's tokenizer. A different model will show a different final dimension — that is expected.

---

## 🟢 Concept Check

**Question 1** — Which layers in a pretrained transformer typically require the least modification during task-specific fine-tuning?

* [x] Early layers encoding syntax and positional patterns
* [ ] Late layers encoding task-specific reasoning
* [ ] The language model head
* [ ] The token embedding matrix

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option A — Early layers encoding syntax and positional patterns.

**Explanation:**
Early transformer layers learn broadly transferable representations of syntax, word order, and positional structure. These generalize well across tasks and typically require little or no updating. Late layers and the LM head are where task-specific behavior concentrates, making them the primary targets for adaptation.
</details>

---

**Question 2** — A 135M-parameter model in `float32` undergoes full fine-tuning with AdamW. Approximately how much memory do the optimizer states alone require?

* [ ] ~135 MB
* [ ] ~540 MB
* [x] ~1,026 MB
* [ ] ~270 MB

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C — ~1,026 MB.

**Explanation:**
AdamW maintains two momentum buffers per parameter (first moment and second moment), each the same size as the weight tensor. For 135M parameters at 4 bytes each: `135M × 4 bytes × 2 = ~1,026 MB`. This is often the single largest memory consumer during full fine-tuning.
</details>

---

**Question 3** — The following code applies LoRA to a model but produces an unexpectedly high trainable parameter count. Identify the most likely cause:

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="all",
)
```

* [ ] `lora_alpha` is set too high relative to `r`.
* [x] Both `r` and `target_modules` are much broader than a minimal LoRA configuration, and `bias="all"` adds all bias terms as trainable.
* [ ] The `task_type` is incorrect for causal language modeling.
* [ ] `r=64` is below the minimum viable rank.

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B.

**Explanation:**
Three factors combine here. First, `r=64` is an 8× larger rank than the `r=8` baseline in this lesson — each adapter matrix grows proportionally. Second, targeting all seven projection layers instead of just `q_proj` and `v_proj` multiplies the adapter count. Third, `bias="all"` makes every bias term in the model trainable, adding parameters that a typical LoRA run excludes. Any one of these decisions alone raises the trainable count; all three together can push it well above 1% of total parameters.
</details>

---

**Question 4 — Reflection** — Describe a real project where an SLM fine-tuned with LoRA would outperform a large hosted LLM API. Justify your reasoning using at least one of these factors: inference latency, data privacy, or domain-scope constraints.

*(Open-ended — no single correct answer.)*

<details>
<summary>🔑 Click to Reveal Guidance</summary>

**Strong answers typically include:**

- **Latency**: A customer-facing autocomplete tool requiring sub-100ms token generation on-device cannot tolerate round-trip API latency. A LoRA-fine-tuned 135M model running locally eliminates network overhead entirely.
- **Data privacy**: A legal discovery tool processing attorney-client privileged documents cannot send text to a third-party API endpoint. Fine-tuning on-premises with LoRA keeps all data inside the organization's security perimeter.
- **Domain scope**: A radiology report normalization task has a narrow vocabulary and structured output format. A large general-purpose LLM may generalize poorly to specialized abbreviations, while a compact SLM fine-tuned on 10,000 curated radiology reports often reaches higher accuracy on that narrow task.

Your answer should name a specific domain, a concrete constraint, and explain *why* the SLM configuration addresses that constraint better than the alternative.
</details>

---

## Summary

- **Pretrained weights encode layered linguistic knowledge** — syntax in early layers, semantics in middle layers, task-specific patterns in late layers. Fine-tuning reuses this knowledge rather than rebuilding it.
- **LoRA reduces GPU memory by roughly 4× for a 135M model** — from ~2.1 GB under full fine-tuning to ~520 MB by limiting trainable parameters to adapter matrices in targeted projection layers.
- **Targeting `q_proj` and `v_proj` with a low rank (r=8) is a reliable starting configuration** — it captures the most task-sensitive attention behavior while keeping the adapter footprint under 2 MB.
- **A verified Module 3 checkpoint is the required entry point** — always run the checkpoint gate check before beginning any LoRA training run to avoid silent initialization errors.

---

## References & Credits

- Hu et al. (2021) *LoRA: Low-Rank Adaptation of Large Language Models*. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
  — First introduced the rank-decomposition adapter approach used throughout this module. Citation applies at first mention of LoRA in Core Concepts.

- Brown et al. (2020) *Language Models are Few-Shot Learners*. [https://arxiv.org/abs/2005.