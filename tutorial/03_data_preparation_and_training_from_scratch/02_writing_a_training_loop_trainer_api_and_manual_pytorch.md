# Writing a Training Loop: Trainer API and Manual PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-seth/slm-development/blob/main/tutorial/03_data_preparation_and_training_from_scratch/02_writing_a_training_loop.ipynb)

| | |
|---|---|
| **Domain** | GenAI |
| **Module** | Data Preparation and Training from Scratch |
| **Difficulty** | Beginner |
| **Estimated Time** | 40 minutes |
| **Prerequisites** | Basic Python programming knowledge; familiarity with what a model is and what training vs. <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr> means; Module 3, Lesson 1 completed (<abbr title="The preprocessing step of converting raw text input into numerical tokens that a language model can process.">tokenization</abbr> and dataset splits); no prior deep learning or NLP experience required |

---

## Lesson Roadmap

- **🟢 Core Concepts** — Understand what a training loop does and where Trainer fits versus raw PyTorch. *(Non-engineers: focus here.)*
- **🔷 Technical Deep-Dive** — Configure `TrainingArguments`, wire up `Trainer`, then write a minimal manual loop. *(Engineers: this is your primary track.)*
- **💡 Compute Budget Planning** — Estimate token throughput and per-step cost before you commit <abbr title="Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning.">GPU</abbr> hours.
- **🧪 Hands-On Exercise** — Train SmolLM2-135M on a custom recipe-instruction dataset and measure <abbr title="A metric measuring how well a probability model predicts a sample; lower perplexity indicates higher confidence and quality.">perplexity</abbr>.
- **✅ Concept Check** — Validate your understanding with scenario-based questions before moving to Module 4.

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Configure a `TrainingArguments` object with learning rate, batch size, and evaluation strategy.
- Train a SmolLM2-135M model on a custom dataset using the Hugging Face `Trainer`.
- Write a minimal manual PyTorch training loop to understand what `Trainer` abstracts away.
- Compute perplexity on a held-out validation set and interpret the result.
- Estimate token throughput and approximate training-step cost before launching a run.

---

## 🟢 Core Concepts

### What a Training Loop Actually Does

Every training run — regardless of framework — repeats the same four-step cycle:

```
┌────────────────────────────────────────────────────────────┐
│  1. FORWARD PASS   → model reads tokens, produces logits   │
│  2. LOSS           → compare logits to ground-truth tokens │
│  3. BACKWARD PASS  → compute gradients via backprop        │
│  4. OPTIMIZER STEP → nudge weights in the right direction  │
└────────────────────────────────────────────────────────────┘
        ↑____________________ repeat per batch ______________|
```

For language modeling the loss function is **cross-entropy**, which measures how surprised the model is by the actual next token. Lower surprise = lower loss = better model. Perplexity is simply `exp(cross-entropy loss)` — a more interpretable number: a perplexity of 20 means the model is as confused as if it were choosing uniformly among 20 equally likely tokens at every step.

### Trainer API vs. Manual Loop — an Analogy

Think of `Trainer` as a commercial oven with preset programmes. You set the temperature (learning rate), the timer (number of epochs), and press start. For most beginner and intermediate runs, this is exactly right.

A manual PyTorch loop is the equivalent of cooking over an open flame — every variable is in your hands. You gain fine-grained control: custom gradient clipping, non-standard schedulers, mid-epoch metric logging. You also gain every possible way to make a mistake.

This lesson shows you both, so you can choose intelligently.

### The SmolLM2-135M Model Family

SmolLM2-135M ([HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), *last verified: 2025-06*) is a 135-million-parameter decoder-only language model designed for on-device and resource-constrained training. At 135 M parameters it fits comfortably in 1–2 GB of GPU <abbr title="Video Random Access Memory: high-speed memory on a GPU used to store model weights and activations during run time.">VRAM</abbr> in full precision, making it a realistic training target for a single consumer GPU or a free-tier Colab session.

> [!IMPORTANT]
> SmolLM2 uses the **LlamaTokenizer** family internally. When you set `labels = input_ids` for causal language modeling, the tokenizer's `pad_token` must equal its `eos_token`. The code below handles this explicitly.

### Compute Budget Planning

Before you run a single training step, estimate whether your hardware can finish the job in reasonable time.

**Key formula:**

```
training_steps = (dataset_tokens / (batch_size × seq_len)) × num_epochs
time_estimate  = training_steps / token_throughput_per_second
```

| Hardware | Approx. token throughput (SmolLM2-135M, bf16) |
|---|---|
| NVIDIA T4 (Colab free) | ~18,000 tokens/s |
| NVIDIA A10G (Colab Pro) | ~55,000 tokens/s |
| Apple M2 (MPS, fp32) | ~6,000 tokens/s |

For this lesson's exercise dataset (~2 M tokens, 3 epochs, batch 8, seq 512): expect **~8 minutes on a T4** and ~25 minutes on an M2. Check your estimate before committing to a longer run.

> [!NOTE]
> <abbr title="A sub-word unit, word, or character that text is split into for processing by a language model.">Token</abbr> throughput degrades roughly linearly as sequence length grows beyond 512 for models without sliding-window attention. SmolLM2-135M supports up to 2048 tokens.

---

## 🔷 Technical Deep-Dive

### Environment Check

```bash
# Verify your environment before starting (Python 3.11 + CUDA 12.1 recommended)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
# Expected: transformers >= 4.40.0, torch >= 2.2.0
```

### Step 1 — Load the Model and Tokenizer

```python
# src/training/load_model.py
# SmolLM2-135M: HuggingFaceTB/SmolLM2-135M (verified 2025-06)

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

def load_smollm2(device: str = "auto") -> tuple:
    """
    Load SmolLM2-135M tokenizer and model.
    Sets pad_token = eos_token, required for causal LM training.
    Returns (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # SmolLM2 has no dedicated pad token — align it with eos_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )

    # Resize embeddings in case pad_token was newly added to the vocabulary.
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


if __name__ == "__main__":
    tok, mdl = load_smollm2()
    total_params = sum(p.numel() for p in mdl.parameters()) / 1e6
    print(f"Loaded {MODEL_ID} — {total_params:.1f}M parameters")
```

### Step 2 — Prepare a Tokenized Dataset

```python
# src/training/prepare_dataset.py
# Uses a fictional recipe-instruction dataset stored locally as JSONL.
# Schema per line: {"instruction": str, "response": str}

from datasets import load_dataset
from transformers import PreTrainedTokenizer

MAX_SEQ_LEN = 512


def tokenize_recipe_dataset(
    jsonl_path: str,
    tokenizer: PreTrainedTokenizer,
    val_split: float = 0.1,
) -> dict:
    """
    Load a JSONL recipe dataset, concatenate instruction + response,
    tokenize with truncation, and produce train/validation splits.
    Returns a DatasetDict with 'train' and 'validation' keys.
    """
    raw = load_dataset("json", data_files={"full": jsonl_path}, split="full")

    def _format_and_tokenize(batch):
        # Merge instruction and response into a single training document.
        texts = [
            f"### Recipe Instruction\n{instr}\n\n### Response\n{resp}"
            for instr, resp in zip(batch["instruction"], batch["response"])
        ]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors=None,  # return plain Python lists for HF datasets
        )
        # For causal LM: labels mirror input_ids; pad positions are masked to -100
        # so cross-entropy ignores them.
        labels = [
            [token_id if token_id != tokenizer.pad_token_id else -100
             for token_id in seq]
            for seq in encoded["input_ids"]
        ]
        encoded["labels"] = labels
        return encoded

    tokenized = raw.map(
        _format_and_tokenize,
        batched=True,
        remove_columns=raw.column_names,
    )
    tokenized.set_format("torch")

    split = tokenized.train_test_split(test_size=val_split, seed=42)
    return {"train": split["train"], "validation": split["test"]}
```

### Step 3 — Train with Hugging Face Trainer (Recommended)

```python
# src/training/train_with_trainer.py

import os
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from load_model import load_smollm2
from prepare_dataset import tokenize_recipe_dataset

# ----- Configuration -----
DATASET_PATH = "data/recipes.jsonl"
OUTPUT_DIR   = "outputs/smollm2-recipe-trainer"
LOG_DIR      = os.path.join(OUTPUT_DIR, "logs")

def build_training_args() -> TrainingArguments:
    """
    TrainingArguments controls every aspect of the training loop.
    Key choices documented inline.
    """
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,

        # --- Learning dynamics ---
        num_train_epochs=3,
        learning_rate=3e-4,          # Common starting point for small LMs from scratch
        lr_scheduler_type="cosine",  # Cosine decay prevents sharp loss spikes late in training
        warmup_ratio=0.05,           # 5% of steps used for linear warmup

        # --- Batch sizing ---
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch = 8 × 4 = 32 samples

        # --- Evaluation & checkpointing ---
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # --- Precision & performance ---
        bf16=True,                   # Use bfloat16 on supported GPUs; fallback handled below
        dataloader_num_workers=2,
        report_to="none",            # Disable W&B/TensorBoard for this exercise

        # --- Logging ---
        logging_dir=LOG_DIR,
        logging_steps=50,
    )


def main():
    tokenizer, model = load_smollm2()
    datasets = tokenize_recipe_dataset(DATASET_PATH, tokenizer)

    # DataCollatorForLanguageModeling handles dynamic padding within each batch.
    # mlm=False → causal (next-token) language modeling, not masked LM.
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = build_training_args()

    # Disable bf16 if CUDA is unavailable (e.g., Apple Silicon / CPU fallback)
    import torch
    if not torch.cuda.is_available():
        training_args.bf16 = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
    )

    trainer.train()

    # Persist the final checkpoint and tokenizer together
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Checkpoint saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

> [!NOTE]
> `gradient_accumulation_steps=4` lets you simulate a batch size of 32 on a GPU that can only hold 8 samples at once. This is one of the most practical knobs available on memory-constrained hardware.

---

### Interpreting Perplexity

| Perplexity Range | What It Signals |
|---|---|
| > 200 | Model has barely started learning; loss is near random |
| 50 – 200 | Early convergence; the model recognises some patterns |
| 15 – 50 | Reasonable for a domain-specific 135M model after a few epochs |
| < 15 | Strong fit; verify you are not <abbr title="A training error where a model learns training data details too well, performing poorly on new data.">overfitting</abbr> the training set |

After 3 epochs on a well-prepared recipe dataset (~2 M tokens), expect SmolLM2-135M to land between 20 and 40 perplexity. A validation perplexity significantly lower than training perplexity is unusual and warrants checking for data leakage between splits.

> [!IMPORTANT]
> <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">Fine-tuning</abbr> (covered in Module 4) typically reaches lower perplexity faster because <abbr title="A model trained on a massive general dataset to learn language patterns before fine-tuning.">pre-trained</abbr> weights already encode language structure. Training from scratch requires more data and more epochs to achieve comparable results.

---

## Hands-On Exercise

**Goal:** Train SmolLM2-135M for 1 epoch on a small sample dataset and verify the perplexity drops below 100.

### Setup (5 minutes)

```bash
# 1. Install dependencies (skip if your devcontainer is active)
pip install transformers==4.44.0 datasets==2.20.0 torch accelerate --quiet

# 2. Create a minimal JSONL sample file for testing
python - <<'EOF'
import json, pathlib, random

INSTRUCTIONS = [
    "How do I make sourdough bread?",
    "What is a quick pasta carbonara recipe?",
    "Describe how to prepare a classic French omelette.",
    "List steps to brew pour-over coffee.",
    "How do I make a simple tomato sauce from scratch?",
]
RESPONSES = [
    "Mix flour, water, salt, and starter. Ferment overnight. Shape, proof, and bake at 230°C for 35 minutes.",
    "Cook guanciale, whisk eggs and pecorino, combine off heat with pasta water to emulsify.",
    "Beat eggs, season, melt butter, pour in pan, fold gently while still slightly wet in the centre.",
    "Grind medium-coarse. Bloom with 30g water for 45s. Pour remainder in slow circles. Total brew: 3 minutes.",
    "Sauté garlic in olive oil, add crushed tomatoes, season, simmer 20 minutes, finish with basil.",
]

pathlib.Path("data").mkdir(exist_ok=True)
rows = [{"instruction": i, "response": r} for i, r in zip(INSTRUCTIONS, RESPONSES)] * 40
random.shuffle(rows)
with open("data/recipes.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print(f"Wrote {len(rows)} records to data/recipes.jsonl")
EOF
```

### Run the Trainer (15 minutes)

```python
# quick_train.py — a self-contained single-file version for the exercise
# Paste this into a Jupyter cell or run as: python quick_train.py

import math, os, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)

MODEL_ID     = "HuggingFaceTB/SmolLM2-135M"
DATASET_PATH = "data/recipes.jsonl"
OUTPUT_DIR   = "outputs/smollm2-exercise"
MAX_SEQ_LEN  = 256  # Shorter for the exercise to reduce runtime

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.resize_token_embeddings(len(tokenizer))

# --- Tokenize ---
raw = load_dataset("json", data_files={"full": DATASET_PATH}, split="full")

def tokenize(batch):
    texts = [
        f"### Instruction\n{i}\n\n### Response\n{r}"
        for i, r in zip(batch["instruction"], batch["response"])
    ]
    enc = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
    enc["labels"] = [
        [t if t != tokenizer.pad_token_id else -100 for t in seq]
        for seq in enc["input_ids"]
    ]
    return enc

tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
tokenized.set_format("torch")
split = tokenized.train_test_split(test_size=0.1, seed=42)

# --- Training arguments ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=3e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    bf16=torch.cuda.is_available(),
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

# --- Evaluate ---
eval_results = trainer.evaluate()
val_loss = eval_results["eval_loss"]
perplexity = math.exp(val_loss)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Perplexity: {perplexity:.2f}")
```

---

## ✅ Concept Check

1. **Why do we align `tokenizer.pad_token` with `tokenizer.eos_token` for causal language modeling?**
   - *Answer:* Causal language models do not have a default pad token like masked language models. Aligning it with the end-of-sequence (EOS) token prevents errors during batch processing and ensures the model ignores padded positions correctly when labels are masked to -100.

2. **What does `gradient_accumulation_steps=4` accomplish?**
   - *Answer:* It accumulates gradients over 4 forward/backward passes before updating weights, effectively multiplying the batch size by 4 without increasing the peak memory footprint (VRAM usage) on the GPU.

3. **If your validation loss is 3.4, what is the validation perplexity?**
   - *Answer:* Perplexity is $e^{3.4} \approx 30$.

---

## 🎓 Confidence Checklist

Before moving on to Module 4, verify that you are confident with the following skills:

- [ ] **Configure** `TrainingArguments` with learning rate, batch size, and weight decay.
- [ ] **Initialize** a Hugging Face `Trainer` to run model training and evaluation.
- [ ] **Implement** gradient accumulation to manage GPU VRAM constraints.
- [ ] **Calculate** and interpret model perplexity from cross-entropy loss.
- [ ] **Estimate** token throughput and training time based on hardware performance.

If you can check all of these, you are ready for Module 4!

---

## 💡 Appendix: Manual PyTorch Training Loop (Optional)

This stripped-down loop mirrors exactly what `Trainer` does internally. Reading it once will make every `TrainingArguments` parameter click.

```python
# src/training/train_manual_loop.py
# A minimal but complete PyTorch training loop for causal language modeling.

import math
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from load_model import load_smollm2
from prepare_dataset import tokenize_recipe_dataset

DATASET_PATH = "data/recipes.jsonl"
NUM_EPOCHS   = 3
BATCH_SIZE   = 8
LR           = 3e-4
WARMUP_RATIO = 0.05
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def compute_perplexity(avg_eval_loss: float) -> float:
    """Perplexity = exp(mean cross-entropy loss). Lower is better."""
    return math.exp(avg_eval_loss)


def evaluate(model, dataloader, device: str) -> tuple[float, float]:
    """
    Run one full pass over the validation set.
    Returns (average_loss, perplexity).
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    return avg_loss, compute_perplexity(avg_loss)


def train():
    tokenizer, model = load_smollm2(device=DEVICE)
    model = model.to(DEVICE)

    datasets = tokenize_recipe_dataset(DATASET_PATH, tokenizer)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            # 1. Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss  # Cross-entropy over non-masked token positions

            # 2. Backward pass
            loss.backward()

            # 3. Gradient clipping — prevents exploding gradients on unstable batches
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 4. Optimizer + scheduler step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch} | Step {global_step}/{total_steps} "
                    f"| Loss: {loss.item():.4f} | LR: {current_lr:.2e}"
                )

        # Evaluate at the end of each epoch
        val_loss, ppl = evaluate(model, val_loader, DEVICE)
        avg_train_loss = epoch_loss / len(train_loader)
        print(
            f"\n=== Epoch {epoch} Summary ===\n"
            f"  Train Loss : {avg_train_loss:.4f}\n"
            f"  Val Loss   : {val_loss:.4f}\n"
            f"  Perplexity : {ppl:.2f}\n"
        )

    print("Manual training loop complete.")


if __name__ == "__main__":
    train()
```
---

## 📝 Chapter Quiz

**Question 1:** What is a defining characteristic of Small Language Models (SLMs) in relation to 02 Writing A Training Loop Trainer Api And Manual Pytorch?

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
