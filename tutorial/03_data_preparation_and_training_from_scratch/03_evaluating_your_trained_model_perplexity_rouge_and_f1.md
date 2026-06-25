# Evaluating Your Trained Model: Perplexity, ROUGE, and F1

| | |
|---|---|
| **Domain** | GenAI |
| **Module** | Data Preparation and Training from Scratch |
| **Difficulty** | Beginner |
| **Estimated Time** | 25 minutes |
| **Prerequisites** | Basic Python programming knowledge; familiarity with what a model is and the difference between training and inference; no prior deep learning or NLP experience required. Complete Module 3 Lessons 1–2 before this lesson. |

---

## Lesson Roadmap

- **Core Concepts** — understand perplexity, ROUGE, and F1 through plain-language analogies before touching any code.
- **Technical Deep-Dive** — compute all three metrics against a real held-out test set using the `evaluate` library (~10 min in).
- **Worked Examples** — walk through a ROUGE-L and F1 calculation step by step with concrete token-level output.
- **Hands-On Exercise** — run the full evaluation pipeline on your own checkpoint and read the results.
- **Concept Check & Failure Mode Diagnosis** — identify overfitting, underfitting, and distribution mismatch from loss curves and metric values.

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Evaluate a trained model using perplexity, ROUGE-L, and F1 metrics on a held-out test set.
- Use the Hugging Face `evaluate` library to compute ROUGE scores on a generation task.
- Interpret a worked ROUGE and F1 example end-to-end, from raw tokens to final score.
- Identify common failure modes — overfitting, catastrophic forgetting, and distribution mismatch — from loss curves and metric values.

---

## 🟢 Core Concepts

### What Are We Actually Measuring?

Training loss tells you how well your model fits the training data. It tells you almost nothing about whether the model is *useful*. You need a separate set of metrics — applied to data the model has never seen — to answer that question.

Three metrics dominate small language model evaluation:

| Metric | Answers the Question | Best Used For |
|---|---|---|
| **Perplexity** | How surprised is the model by real text? | Language modeling quality |
| **ROUGE** | How much does the output overlap the reference? | Summarization, text generation |
| **F1** | How balanced is precision vs. recall at the token level? | QA, extraction, classification |

---

### Perplexity: The Surprise Score

Perplexity (PPL) measures how confidently your model assigns probability to real text. A model with PPL of 10 assigns, on average, roughly equal probability to 10 candidate tokens at each position. Lower is better.

```
PPL = exp(average negative log-likelihood per token)
```

A few anchors for calibration:

- **PPL < 20** on domain-specific text: generally good for a small model.
- **PPL > 100**: the model is nearly guessing — check your tokenizer alignment and training data quality.
- **PPL drops on train but rises on validation**: classic overfitting signal.

> [!IMPORTANT]
> Always compute perplexity on your **held-out test split**, never on training data. A model that has memorized training examples will show artificially low PPL on that data.

---

### ROUGE: Overlap-Based Scoring

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) counts n-gram overlaps between a model's output and a human-written reference.

- **ROUGE-1**: unigram (single-word) overlap.
- **ROUGE-2**: bigram (two-word sequence) overlap.
- **ROUGE-L**: longest common subsequence — rewards fluent, ordered output even with gaps.

**Worked Example — Scoring a Recipe Summary**

```
Reference : "simmer the tomatoes and garlic until soft"
Hypothesis: "cook the tomatoes with garlic until tender"
```

Matching unigrams: `the`, `tomatoes`, `garlic`, `until` → 4 of 7 reference tokens matched.

```
ROUGE-1 Recall    = 4/7  ≈ 0.57
ROUGE-1 Precision = 4/8  ≈ 0.50
ROUGE-1 F1        = 2 × (0.57 × 0.50)/(0.57 + 0.50) ≈ 0.53
```

ROUGE-L looks for the longest subsequence both strings share in order: `tomatoes garlic until` (length 3).

```
ROUGE-L Recall    = 3/7 ≈ 0.43
ROUGE-L Precision = 3/8 ≈ 0.38
ROUGE-L F1        ≈ 0.40
```

---

### F1: Precision and Recall in Balance

For token-level tasks (extractive QA, entity extraction), F1 captures both whether you produced the right tokens (precision) and whether you missed any (recall).

```
Precision = correct_tokens / predicted_tokens
Recall    = correct_tokens / reference_tokens
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

An F1 of 1.0 means perfect overlap. An F1 of 0.0 means no shared tokens at all.

> [!NOTE]
> ROUGE and F1 are surface-level metrics. A semantically correct paraphrase ("tender" for "soft") will score lower than a verbatim match. Use them as signals, not verdicts.

---

## 🔷 Technical Deep-Dive

### Environment Setup

```bash
# Requires Python 3.11, PyTorch 2.2+, CUDA 12.1 (or CPU fallback)
pip install "transformers==4.41.2" \
            "evaluate==0.4.2" \
            "datasets==2.19.2" \
            "rouge_score==0.1.2" \
            "torch==2.2.2" \
            --quiet
# Last verified: 2025-06
```

> [!NOTE]
> If you haven't set up your training environment yet, complete Module 3 Lesson 1 first. That lesson walks through virtual environment creation and dependency pinning.

---

### Step 1 — Load Your Checkpoint and Test Split

This example uses a SmolLM2-135M checkpoint fine-tuned on a cooking instruction dataset. Swap `CHECKPOINT_DIR` for your own path.

```python
# evaluate_model.py
# PEP 8 compliant | no hardcoded secrets | meaningful variable names

import os
import torch
import evaluate
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configuration ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./outputs/smollm2-cooking-v1")
TEST_DATASET_DIR = os.environ.get("TEST_DATASET_DIR", "./data/cooking_test")
MAX_NEW_TOKENS = 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Loading checkpoint from: {CHECKPOINT_DIR}")

# ── Model & tokenizer ──────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_DIR,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
model.to(DEVICE)
model.eval()

# ── Test dataset ───────────────────────────────────────────────────────────────
test_dataset = load_from_disk(TEST_DATASET_DIR)
print(f"[INFO] Test set size: {len(test_dataset)} examples")
```

---

### Step 2 — Compute Perplexity

```python
# evaluate_model.py (continued)

def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str,
    stride: int = 512,
) -> float:
    """
    Compute mean perplexity over a list of texts using a sliding-window
    approach to handle sequences longer than the model's context window.
    """
    total_log_likelihood = 0.0
    total_token_count = 0

    for text in texts:
        encoded = tokenizer(text, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]
        seq_len = input_ids.size(1)

        # Slide over long sequences to avoid OOM on small GPUs
        for start in range(0, seq_len, stride):
            end = min(start + stride, seq_len)
            chunk = input_ids[:, start:end]

            with torch.no_grad():
                outputs = model(chunk, labels=chunk)

            # outputs.loss is mean NLL over this chunk
            chunk_nll = outputs.loss.item() * (end - start)
            total_log_likelihood += chunk_nll
            total_token_count += (end - start)

    mean_nll = total_log_likelihood / total_token_count
    perplexity = torch.exp(torch.tensor(mean_nll)).item()
    return round(perplexity, 4)


sample_texts = [row["text"] for row in test_dataset.select(range(50))]
ppl = compute_perplexity(model, tokenizer, sample_texts, DEVICE)
print(f"[RESULT] Perplexity on test set (n=50): {ppl}")
```

**Expected output for a reasonably trained SmolLM2-135M:**
```
[RESULT] Perplexity on test set (n=50): 18.34
```

---

### Step 3 — Generate Predictions and Compute ROUGE

```python
# evaluate_model.py (continued)

rouge_metric = evaluate.load("rouge")

def generate_summary(prompt: str) -> str:
    """Generate a short continuation given a prompt string."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens, not the prompt
    new_token_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


# Build prediction / reference lists from the test set
predictions: list[str] = []
references: list[str] = []

# Use first 30 examples to keep runtime under 5 minutes on CPU
eval_subset = test_dataset.select(range(30))

for example in eval_subset:
    prompt = example["prompt"]           # e.g., "Summarize: Roast the peppers..."
    reference_text = example["target"]   # human-written reference

    prediction = generate_summary(prompt)
    predictions.append(prediction)
    references.append(reference_text)

rouge_scores = rouge_metric.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True,       # normalizes "cooking" / "cook" as equivalent
)

print("\n[RESULT] ROUGE Scores:")
for key, value in rouge_scores.items():
    print(f"  {key}: {value:.4f}")
```

**Example output:**
```
[RESULT] ROUGE Scores:
  rouge1: 0.4812
  rouge2: 0.2934
  rougeL: 0.4103
```

---

### Step 4 — Compute Token-Level F1

The `evaluate` library ships an `exact_match` and `squad` metric for QA-style F1. For a generative task, compute token-level F1 manually:

```python
# evaluate_model.py (continued)

from collections import Counter


def token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 between two strings.
    Tokenization is whitespace-based; case-insensitive.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Intersection: count tokens that appear in both
    shared = sum((pred_counter & ref_counter).values())

    if shared == 0:
        return 0.0

    precision = shared / len(pred_tokens)
    recall = shared / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


f1_scores = [
    token_f1(pred, ref)
    for pred, ref in zip(predictions, references)
]
mean_f1 = sum(f1_scores) / len(f1_scores)
print(f"\n[RESULT] Mean Token-Level F1 (n=30): {mean_f1:.4f}")
```

**Example output:**
```
[RESULT] Mean Token-Level F1 (n=30): 0.5217
```

---

### Step 5 — Reading Loss Curves to Diagnose Failure Modes

Plot training vs. validation loss after training:

```python
# plot_loss_curves.py

import json
import matplotlib.pyplot as plt
from pathlib import Path

TRAINER_STATE_PATH = os.environ.get(
    "TRAINER_STATE_PATH",
    "./outputs/smollm2-cooking-v1/trainer_state.json",
)

state_path = Path(TRAINER_STATE_PATH)
if not state_path.exists():
    raise FileNotFoundError(f"trainer_state.json not found at {state_path}")

with state_path.open() as f:
    trainer_state = json.load(f)

log_history = trainer_state["log_history"]

train_steps, train_losses = [], []
eval_steps, eval_losses = [], []

for entry in log_history:
    if "loss" in entry:
        train_steps.append(entry["step"])
        train_losses.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_losses.append(entry["eval_loss"])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_steps, train_losses, label="Train Loss", linewidth=2)
ax.plot(eval_steps, eval_losses, label="Validation Loss", linewidth=2, linestyle="--")
ax.set_xlabel("Training Step")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Training vs. Validation Loss — SmolLM2-135M Cooking Fine-Tune")
ax.legend()
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
print("[INFO] Loss curve saved to loss_curves.png")
```

**How to read the output:**

```
Scenario A — Healthy training
  Train loss ↓  Validation loss ↓ (tracking closely)
  → Continue training or stop at the validation minimum.

Scenario B — Overfitting
  Train loss ↓  Validation loss ↓ then ↗ (diverges upward)
  → Apply early stopping, increase dropout, or reduce epochs.

Scenario C — Underfitting
  Both losses remain high and flat
  → Check learning rate (often too low), data quality, or batch size.

Scenario D — Catastrophic forgetting
  Validation loss on a *general* benchmark rises sharply
  after fine-tuning on domain data
  → Use LoRA / PEFT to freeze base weights, or mix in general-domain data.
```

> [!IMPORTANT]
> Catastrophic forgetting is a fine-tuning risk, not a training-from-scratch risk. If you trained from scratch, Scenario D is more likely distribution mismatch: your test data comes from a different source than your training corpus.

---

## Hands-On Exercise

**Goal:** Run the full evaluation pipeline on your Module 3 Lesson 2 checkpoint and record your results.

### Step 1 — Export environment variables

```bash
export CHECKPOINT_DIR="./outputs/your-model-checkpoint"
export TEST_DATASET_DIR="./data/your_test_split"
export TRAINER_STATE_PATH="./outputs/your-model-checkpoint/trainer_state.json"
```

### Step 2 — Run evaluation

```bash
python evaluate_model.py
```

Record the three numbers printed:
```
Perplexity  : ____
ROUGE-L     : ____
Mean F1     : ____
```

### Step 3 — Plot loss curves

```bash
python plot_loss_curves.py
open loss_curves.png   # macOS
# xdg-open loss_curves.png   # Linux
```

### Step 4 — Diagnose

Using the four scenarios from the Deep-Dive, classify your training run:

| Observation | Your Value | Scenario |
|---|---|---|
| Train loss trend | ↓ / flat / ↑ | |
| Validation loss trend | ↓ / flat / diverges | |
| ROUGE-L vs. baseline (0.3) | above / below | |
| Perplexity vs. 20 threshold | above / below | |

**Verifiable outcome:** You can state whether your model is overfitting, underfitting, or training healthily — backed by specific metric values.

> [!NOTE]
> If your test split is smaller than 30 examples, reduce the `range(30)` in Step 3 of `evaluate_model.py` to match your actual test set size.

---

## Concept Check

**Question 1**

Your training loss drops steadily from 3.2 to 1.1 over 5 epochs. Your validation loss drops to 1.4 at epoch 3, then climbs to 2.1 by epoch 5. Which failure mode does this describe?

* [ ] Underfitting — the model cannot learn the training data.
* [x] Overfitting — the model has memorized training examples and generalizes poorly.
* [ ] Distribution mismatch — the test data comes from a different domain.
* [ ] Catastrophic forgetting — prior knowledge has been erased.

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B — Overfitting.

**Explanation:**
The training loss improves continuously while the validation loss initially falls and then rises. This divergence is the textbook overfitting signature. The fix is early stopping at epoch 3 (the validation minimum), reducing the learning rate, or adding regularization such as dropout or weight decay. Distribution mismatch would show high validation loss *from the start*, not a rise after an initial drop.

</details>

---

**Question 2**

A worked ROUGE-1 calculation gives Recall = 0.70, Precision = 0.50. What is the ROUGE-1 F1 score, rounded to two decimal places?

* [ ] 0.60
* [ ] 0.35
* [x] 0.58
* [ ] 0.70

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C — 0.58.

**Explanation:**
ROUGE-1 F1 = 2 × (Precision × Recall) / (Precision + Recall)  
= 2 × (0.50 × 0.70) / (0.50 + 0.70)  
= 2 × 0.35 / 1.20  
= 0.70 / 1.20  
≈ **0.583**, which rounds to 0.58.

The harmonic mean penalizes imbalanced precision/recall more than a simple average would, which is why the result (0.58) falls below the arithmetic mean (0.60).

</details>

---

**Question 3 — Open-Ended Reflection**

Your model achieves a ROUGE-L of 0.21 on a cooking summarization task. A teammate suggests re-running training with 10× more data. Before agreeing, what two alternative diagnoses would you check first, and what metric or artifact would you inspect for each?

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Suggested reasoning (your answer may differ):**

1. **Distribution mismatch** — Check whether your test set was drawn from the same source as your training set. If you trained on recipe blogs and tested on culinary textbooks, the domain gap will depress ROUGE regardless of data volume. Inspect a few test examples manually; look for vocabulary, sentence length, and style differences.

2. **Decoding configuration** — Check `max_new_tokens` and whether you're using greedy vs. beam search. If `max_new_tokens` is too low, the model truncates outputs before completing a reference-length summary, which mechanically lowers recall and thus ROUGE. Run `generate_summary` on a single example and print the raw output before computing scores.

More data helps only when the fundamental problem is data scarcity. Ruling out mismatch and decoding issues first saves significant compute time.

</details>

---

## Summary

- **Perplexity** measures how surprised your model is by held-out text; values below ~20 signal healthy language modeling for a small domain-specific model.
- **ROUGE** (particularly ROUGE-L) quantifies n-gram and sequence overlap between generated output and a reference; it is the standard metric for summarization and text generation tasks.
- **Token-level F1** balances precision and recall on exact token matches; use it alongside ROUGE for extraction and QA tasks where missing or adding tokens carries different costs.
- **Loss curves** are your first diagnostic tool: training loss falling while validation loss rises signals overfitting; both staying flat signals underfitting; a sharp validation spike after domain fine-tuning signals catastrophic forgetting.
- The Hugging Face `evaluate` library (`evaluate.load("rouge")`) handles ROUGE computation in three lines of code — no manual n-gram counting required.
- Surface-level metrics like ROUGE and F1 do not capture semantic correctness; treat them as signals, then inspect model outputs manually before drawing conclusions.

---

## References & Credits

- **ROUGE**: Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. Proceedings of the Workshop on Text Summarization Branches Out, ACL 2004. [https://aclanthology.org/W04-1013](https://aclanthology.org/W04-1013)
- **BPE Tokenization**: Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*. [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
- **Catastrophic Forgetting**: McCloskey, M., & Cohen, N. J. (1989). *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem*. Psychology of Learning and Motivation, 24, 109–165. [https://doi.org/10.1016/S0079-7421(08)60536-8](https://doi.org/10.1016/S0079-7421(08)60536-8)
- **LoRA (PEFT)**: Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685