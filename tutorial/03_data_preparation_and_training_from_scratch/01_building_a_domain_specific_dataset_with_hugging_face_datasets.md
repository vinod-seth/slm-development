# Building a Domain-Specific Dataset with Hugging Face Datasets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-seth/slm-development/blob/main/tutorial/03_data_preparation_and_training_from_scratch/01_building_a_domain_specific_dataset.ipynb)

| | |
|---|---|
| **Domain** | GenAI |
| **Module** | Data Preparation and Training from Scratch |
| **Difficulty** | Beginner |
| **Estimated Time** | 30 minutes |
| **Prerequisites** | Basic Python programming knowledge; familiarity with what a model is and the difference between training and <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr>; no prior deep learning or NLP experience required |

---

## Lesson Roadmap

- **Core Concepts** — Understand how datasets flow from the Hub into a training pipeline, using a schema-to-tokens mental model.
- **Technical Deep-Dive** — Load a real public dataset, inspect its schema, tokenize it with batched `map()`, and produce reproducible train/validation/test splits.
- **Hands-On Exercise** — Build a complete preprocessing pipeline for a domain-specific corpus drawn from the medical question-answering domain.
- **Concept Check** — Three questions testing schema inspection, <abbr title="The preprocessing step of converting raw text input into numerical tokens that a language model can process.">tokenization</abbr> strategy, and split reproducibility.
- **Summary & References** — Key takeaways and academic citations for any foundational techniques referenced.

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Load a public dataset from the Hugging Face Hub and inspect its schema using `DatasetDict`.
- Write a preprocessing function that tokenizes text with dynamic padding and truncation.
- Apply `dataset.map()` with batched processing to tokenize at scale without memory overflow.
- Split a dataset into train, validation, and test partitions with a reproducible random seed.
- Explain the role of a data collator in causal language modeling.

---

## 🟢 Core Concepts

### Datasets as Pipelines, Not Files

Most developers think of a dataset as a file — a CSV you download and load into memory. The Hugging Face `datasets` library works differently. It treats a dataset as a **lazy, memory-mapped pipeline**: records stream from disk on demand. This matters when your corpus is larger than your available RAM.

The mental model looks like this:

```mermaid
flowchart LR
    A["HF Hub\n(remote storage)"] -->|load_dataset| B["DatasetDict\n(schema + splits)"]
    B -->|.map(tokenize_fn)| C["Tokenized DatasetDict\n(input_ids, attention_mask)"]
    C -->|DataCollator| D["Padded Batches\n(ready for model)"]
    D --> E["Training Loop"]
```

Each arrow is a transformation. No single step forces the entire dataset into RAM at once.

### Schema Inspection: Know Before You Transform

A `DatasetDict` contains named splits — typically `"train"`, `"validation"`, and `"test"`. Each split holds **columns** (fields) and **rows** (examples). Before writing a single tokenization line, inspect the schema. Surprises here — a missing label column, an unexpected text encoding — cause silent failures downstream.

### Tokenization Strategy: Padding vs. Truncation

A tokenizer converts raw text into integer token IDs. Two decisions control output length:

- **Truncation** (`truncation=True`): Clips sequences longer than `max_length`. Essential — models have a fixed context window.
- **Padding**: Adds a special `[PAD]` token so all sequences in a batch share the same length.

For training efficiency, avoid **static padding** (pad every sequence to `max_length`). Use a **data collator** instead, which pads only to the longest sequence *within each batch*. This is called **dynamic padding** and it cuts <abbr title="Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning.">GPU</abbr> memory waste significantly.

### Causal Language Modeling and Data Collators

In causal language modeling (CLM), the model predicts the next token given all previous tokens. The training label for each sequence is the sequence itself, shifted one position right. The `DataCollatorForLanguageModeling` from Hugging Face handles this label construction automatically when you set `mlm=False`.

> [!IMPORTANT]
> The data collator, not the tokenizer, is responsible for final batch padding during training. Pass `padding=False` to your tokenizer inside `map()` to avoid redundant padding before the collator runs.

---

## 🔷 Technical Deep-Dive

> [!NOTE]
> Run the install block below before executing any other cell. If you are new to Python environments, complete Module 3's environment setup (Lesson 3) before proceeding here.

### Step 1 — Install Dependencies

```bash
# Verified against: datasets==2.19.x, transformers==4.41.x, torch==2.3.x
# Last verified: 2025-06
pip install datasets transformers torch --quiet
```

### Step 2 — Load a Public Dataset and Inspect Its Schema

We use [`medmcqa`](https://huggingface.co/datasets/medmcqa), a publicly available multiple-choice medical QA dataset (Apache 2.0). It provides a realistic domain-specific corpus without requiring a Hugging Face account.

```python
# dataset_pipeline.py
# Preprocessing pipeline for domain-specific SLM training.
# Dataset: medmcqa (Apache 2.0) — medical multiple-choice QA

from datasets import load_dataset

# Stream the dataset from the Hub; trust_remote_code not required here.
medical_dataset = load_dataset("medmcqa")

# Inspect the top-level structure.
print(type(medical_dataset))          # <class 'datasets.DatasetDict'>
print(medical_dataset)                # Shows all splits and row counts.

# Inspect column names and data types for the training split.
print("\n--- Schema ---")
print(medical_dataset["train"].features)

# Preview the first record to understand field content.
first_record = medical_dataset["train"][0]
print("\n--- First Record ---")
for field, value in first_record.items():
    print(f"  {field}: {value!r}")
```

**Expected output (abbreviated):**

```
DatasetDict({
    train: Dataset({features: ['id', 'question', 'opa', 'opb', 'opc', 'opd',
                               'cop', 'choice_type', 'exp', 'subject_name',
                               'topic_name'], num_rows: 182822}),
    validation: Dataset({...num_rows: 4183}),
    test: Dataset({...num_rows: 6150})
})
```

> [!NOTE]
> The setup sequence above follows standard Hugging Face installation conventions. The procedural steps reflect common documented practice; any similarity to official HF documentation is coincidental and not reproduced verbatim.

### Step 3 — Build the Text Column

`medmcqa` stores question text and answer options in separate columns. Concatenate them into a single `"text"` field for causal language modeling.

```python
def build_training_text(example: dict) -> dict:
    """
    Merge question and answer options into a single training string.
    Format: 'Q: <question> A: <correct_option_text>'
    """
    option_map = {
        0: example["opa"],
        1: example["opb"],
        2: example["opc"],
        3: example["opd"],
    }
    correct_index = example["cop"]
    correct_answer = option_map.get(correct_index, "")

    merged_text = f"Q: {example['question'].strip()} A: {correct_answer.strip()}"
    return {"text": merged_text}


# Apply the merge function to all splits.
medical_dataset = medical_dataset.map(
    build_training_text,
    batched=False,          # Row-level operation — no batching needed here.
    desc="Building text column",
)

print(medical_dataset["train"]["text"][0])
```

### Step 4 — Tokenize with Batched `map()`

Load a tokenizer appropriate for a small language model. We use `HuggingFaceTB/SmolLM2-135M` (Apache 2.0) — a 135 M-parameter model from the SmolLM2 family, designed for edge and resource-constrained deployment.

```python
from transformers import AutoTokenizer

MODEL_CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"  # Last verified: 2025-06
MAX_SEQUENCE_LENGTH = 256  # Fits comfortably within SmolLM2's 2048-token context.

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# SmolLM2 uses an EOS token but no dedicated PAD token by default.
# Assign EOS as the pad token to satisfy the collator's requirements.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_batch(batch: dict) -> dict:
    """
    Tokenize a batch of text strings.

    - truncation=True: clips sequences exceeding MAX_SEQUENCE_LENGTH.
    - padding=False: defer padding to DataCollatorForLanguageModeling.
    - return_special_tokens_mask=False: not needed for CLM.
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        padding=False,  # Dynamic padding happens in the collator, not here.
    )


tokenized_dataset = medical_dataset.map(
    tokenize_batch,
    batched=True,           # Process 1000 rows at a time (default batch size).
    batch_size=1000,        # Explicit for clarity; tune down if OOM on CPU.
    remove_columns=medical_dataset["train"].column_names,  # Drop raw text columns.
    desc="Tokenizing",
)

print(tokenized_dataset)
print("\nSample input_ids (first 20 tokens):")
print(tokenized_dataset["train"][0]["input_ids"][:20])
```

> [!IMPORTANT]
> `remove_columns=medical_dataset["train"].column_names` drops all original string columns after tokenization. If you skip this, the trainer will attempt to batch-collate raw strings alongside tensors and raise a `TypeError`.

### Step 5 — Create Reproducible Splits

`medmcqa` already ships with validation and test splits. If your custom dataset has only a `"train"` split, use `train_test_split()` with a fixed seed.

```python
# Demonstration: how to split a single-split dataset reproducibly.
# medmcqa already has splits, so we apply this to a subset for illustration.

single_split_sample = tokenized_dataset["train"].select(range(5000))

# First split: carve out 20% as the combined validation+test pool.
initial_split = single_split_sample.train_test_split(
    test_size=0.20,
    seed=42,           # Pin the seed for reproducibility across runs.
    shuffle=True,
)

# Second split: divide the 20% pool into equal validation and test halves.
val_test_split = initial_split["test"].train_test_split(
    test_size=0.50,
    seed=42,
)

from datasets import DatasetDict

reproducible_splits = DatasetDict({
    "train":      initial_split["train"],
    "validation": val_test_split["train"],
    "test":       val_test_split["test"],
})

print(reproducible_splits)
# train: 4000 rows | validation: 500 rows | test: 500 rows
```

### Step 6 — Configure the Data Collator

```python
from transformers import DataCollatorForLanguageModeling

# mlm=False: configure for causal (autoregressive) language modeling.
# The collator shifts labels internally; you do not need to do this manually.
clm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Quick sanity check: collate two samples into a batch.
import torch

sample_batch = [tokenized_dataset["train"][i] for i in range(2)]
collated = clm_collator(sample_batch)

print("Collated keys:", list(collated.keys()))
print("input_ids shape:", collated["input_ids"].shape)
print("labels shape:   ", collated["labels"].shape)
# input_ids and labels share the same shape; labels are shifted inside the model.
```

---

## Hands-On Exercise

**Goal:** Build a complete tokenized dataset from a subset of `medmcqa`, verify the schema at each stage, and confirm that train/validation/test row counts match expected proportions.

**Outcome you can verify:** Running the final `print(splits)` statement produces three splits with row counts matching the 80/10/10 ratio within ±5 rows.

### Steps

1. **Load** the `medmcqa` dataset. Print the feature names of the `"train"` split.

2. **Select** the first 2,000 rows of the training split using `.select(range(2000))`. This keeps runtime under 2 minutes on a <abbr title="Central Processing Unit: the general-purpose processor in a computer.">CPU</abbr>.

3. **Apply** `build_training_text` (from the deep-dive) to create the `"text"` column.

4. **Tokenize** using `tokenize_batch` with `MAX_SEQUENCE_LENGTH = 128`. Confirm `"input_ids"` appears in the output features.

5. **Split** into 80% train, 10% validation, 10% test using two sequential `train_test_split()` calls with `seed=2024`.

6. **Verify** by running:

```python
for split_name, split_data in splits.items():
    print(f"{split_name}: {len(split_data)} rows")
```

Expected output:

```
train:      1600 rows
validation:  200 rows
test:        200 rows
```

7. **Instantiate** `DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)`. Collate rows 0–3 from your train split. Print the shape of `collated["input_ids"]`.

> [!NOTE]
> If you see a `ValueError` about the pad token, confirm you executed `tokenizer.pad_token = tokenizer.eos_token` before calling the collator.

---

## Concept Check

**Question 1**

You run `load_dataset("medmcqa")` and print the result. Which Python type does the returned object have?

* [ ] `list`
* [ ] `pandas.DataFrame`
* [x] `datasets.DatasetDict`
* [ ] `torch.utils.data.Dataset`

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** `datasets.DatasetDict`

**Explanation:**
`load_dataset()` returns a `DatasetDict` when the target dataset ships with multiple named splits (e.g., `"train"`, `"validation"`, `"test"`). Each value in the dict is a `Dataset` object containing rows and typed features. A bare `Dataset` is returned only when you request a single split explicitly, e.g., `load_dataset("medmcqa", split="train")`.

</details>

---

**Question 2**

Your tokenizer call inside `map()` uses `padding="max_length"` instead of `padding=False`. What is the most likely consequence during training?

* [ ] The model raises a `RuntimeError` because padded tokens cannot be processed.
* [x] GPU memory usage increases because sequences are padded to `max_length` before the collator runs, producing unnecessarily large tensors.
* [ ] Training accuracy drops because padded tokens contribute to the loss calculation by default.
* [ ] No consequence — padding strategy has no effect on throughput.

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** GPU memory usage increases.

**Explanation:**
Static padding inside `map()` pads every sequence to `MAX_SEQUENCE_LENGTH` before batching. This inflates tensor sizes even when most sequences in a batch are short. The `DataCollatorForLanguageModeling` is designed to handle padding dynamically — padding only to the longest sequence *in the current batch*. Doing both is redundant and wasteful. Regarding loss: by default, the collator sets label positions corresponding to pad tokens to `-100`, which tells PyTorch's cross-entropy loss to ignore them. So accuracy is not directly harmed, but throughput and memory are.

</details>

---

**Question 3 — Reflection Prompt**

Consider a case where your domain-specific corpus has only 800 labeled examples — far fewer than `medmcqa`'s 182,000 rows. How would you adjust your train/validation/test split ratios, and what risks does a very small validation set introduce to your evaluation of model performance?

* [ ] Keep the 80/10/10 ratio — it is universally correct regardless of dataset size.
* [ ] Use 50/25/25 — larger held-out sets always produce more reliable evaluation.
* [x] Adjust ratios based on absolute row counts; consider k-fold cross-validation when validation sets are too small to be statistically meaningful.
* [ ] Merge validation and test into one set to maximize training data.

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Adjust ratios based on absolute row counts; consider k-fold cross-validation.

**Explanation:**
An 80/10/10 split on 800 examples yields only 80 validation rows. Evaluation metrics computed on 80 samples carry high variance — a single unusual batch can swing your <abbr title="A metric measuring how well a probability model predicts a sample; lower perplexity indicates higher confidence and quality.">perplexity</abbr> estimate by several points. For small corpora, k-fold cross-validation provides more stable estimates by rotating the validation window across the full dataset. Merging validation and test sets is dangerous: the test set should remain unseen until final evaluation; using it during hyperparameter tuning causes data leakage.

**Open-ended extension:** Describe a real project domain (e.g., internal legal Q&A, rare-disease clinical notes) where you would face a sub-1,000-example constraint. What data augmentation or synthetic generation strategies might help, and what risks do those introduce?

</details>

---

## Summary

- `load_dataset()` returns a memory-mapped `DatasetDict`. Inspect `.features` before writing any transformation logic to catch schema surprises early.
- Pass `padding=False` to the tokenizer inside `map()` and let `DataCollatorForLanguageModeling(mlm=False)` handle dynamic padding at batch time. This reduces GPU memory waste.
- Fix your random seed in every `train_test_split()` call. Reproducibility is not optional — it ensures that model comparisons across runs reflect genuine performance differences, not sampling variation.
- `remove_columns` inside `map()` prevents the trainer from attempting to collate raw string columns alongside tensors, which causes silent or hard-to-debug runtime errors.
- For causal language modeling, the collator — not manual label construction — shifts token IDs to build training targets. Trust the abstraction; verify it with a quick shape check.

---

## References & Credits

- **Dataset:** `medmcqa` — Pal et al. (2022) *MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering*. [https://arxiv.org/abs/2203.14371](https://arxiv.org/abs/2203.14371). Licensed under Apache 2.0.
- **Model checkpoint:** `HuggingFaceTB/SmolLM2-135M` — Hugging Face Technology Team (2024). SmolLM2 model family. [https://huggingface.co/HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). Licensed under Apache 2.0. Last verified: 2025-06.
- **Tokenizer BPE foundations** — Sennrich et al. (2016) *Neural Machine Translation of Rare Words with Subword Units*. [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909). *(BPE is the subword algorithm underlying SmolLM2's tokenizer.)*
- **Hugging Face `datasets` library** — Lhoest et al. (2021) *Datasets: A Community Library for Natural Language Processing*. [https://arxiv.org/abs/2109.02846](https://arxiv.org/abs/2109.02846). Licensed under Apache 2.0.
- **<abbr title="Low-Rank Adaptation: an efficient fine-tuning method that freezes base model weights and injects small trainable adapter matrices.">LoRA</abbr> <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">fine-tuning</abbr>** (referenced in later modules) — Hu et al. (2021) *LoRA: Low-Rank Adaptation of Large Language Models*. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685).