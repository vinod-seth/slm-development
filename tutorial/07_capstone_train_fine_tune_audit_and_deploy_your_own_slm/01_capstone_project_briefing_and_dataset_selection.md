# Capstone Project Briefing and Dataset Selection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-seth/slm-development/blob/main/tutorial/07_capstone_train_fine_tune_audit_and_deploy_your_own_slm/01_capstone_briefing.ipynb)

| | |
|---|---|
| **Domain** | GenAI |
| **Module** | Capstone: Train, Fine-Tune, Audit, and Deploy Your Own <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLM</abbr> |
| **Difficulty** | Beginner |
| **Estimated Time** | 20 minutes |
| **Prerequisites** | Basic Python programming knowledge · Familiarity with fundamental ML concepts (training vs. <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr>, what a model is) · No prior deep learning or NLP experience required · Modules 1–6 of this course completed |

---

## Lesson Roadmap

- **Minutes 0–4** — Core Concepts: understand the four task types, compute tiers, and how to scope your domain
- **Minutes 4–8** — Technical Deep-Dive: browse HF Hub, verify dataset licenses, and run a dataset preview in Python
- **Minutes 8–14** — Hands-On Exercise: select your domain, dataset, and compute tier; fill in your project plan template
- **Minutes 14–17** — Concept Check: three questions to confirm you can apply the scoping framework
- **Minutes 17–20** — Summary and reference reading

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Select a target domain and source a suitable public dataset from the Hugging Face Hub
- Define a concrete task (text classification, Q&A, summarization, or code completion) with measurable success criteria
- Outline a project plan covering data preparation, <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">fine-tuning</abbr>, evaluation, bias audit, and deployment steps
- Identify the compute tier available to you and select an appropriate SLM checkpoint accordingly

---

## 🟢 Core Concepts

### The Four Capstone Task Types

Your capstone trains and deploys one small language model on one task. Keeping the scope narrow is not a limitation — it is the lesson. Production SLM deployments almost always target a single well-defined task.

| Task | Input → Output | Example Domain |
|---|---|---|
| **Text Classification** | Raw text → label | Customer support ticket routing |
| **Extractive / Generative Q&A** | Context + question → answer | Internal policy lookup |
| **Summarization** | Long document → short summary | Medical discharge notes |
| **Code Completion** | Partial function → complete function | Developer tooling |

Pick the task that matches a problem you actually care about. A narrow, meaningful task produces a more useful model than a broad, generic one.

---

### Compute Tier Selection

Before choosing a model checkpoint, know your hardware. The table below maps <abbr title="Video Random Access Memory: high-speed memory on a GPU used to store model weights and activations during run time.">VRAM</abbr> to realistic SLM choices verified against current Hugging Face Model Hub listings.

| Tier | Hardware | Recommended Checkpoint | Parameter Range |
|---|---|---|---|
| **<abbr title="Central Processing Unit: the general-purpose processor in a computer.">CPU</abbr>-only** | Any laptop | `HuggingFaceTB/SmolLM2-135M` | 135 M |
| **8 GB VRAM** | Consumer <abbr title="Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning.">GPU</abbr> (RTX 3070 / T4 Colab) | `HuggingFaceTB/SmolLM2-1.7B` | 1.7 B |
| **16 GB VRAM** | Prosumer GPU (RTX 3090 / A10 Colab Pro) | `microsoft/phi-2` | 2.7 B |

> [!IMPORTANT]
> `microsoft/phi-2` requires accepting a license on the Hugging Face Hub before downloading. Create a free HF account and accept the license at `huggingface.co/microsoft/phi-2` before Module 4 fine-tuning steps.

*Last verified: 2025-06*

---

### Dataset License Verification

Using a dataset with a restrictive license in a deployed product creates legal risk. Every dataset on the HF Hub exposes its license in the dataset card. The licenses safe for educational and commercial capstone use are:

- `apache-2.0`
- `mit`
- `cc-by-4.0`
- `cc0-1.0` (public domain)

Avoid `cc-by-nc-*` (non-commercial) if you plan to deploy externally. Avoid datasets with no license listed — treat those as all-rights-reserved.

---

### Project Plan: Five Phases

Your capstone spans five phases. Write one sentence of intent for each before you write a single line of training code.

```
Phase 1 → Data Preparation   (clean, split, tokenize)
Phase 2 → Fine-Tuning        (LoRA / PEFT on chosen checkpoint)
Phase 3 → Evaluation         (task metric: F1, ROUGE, pass@1)
Phase 4 → Bias & Safety Audit (adversarial prompts, failure documentation)
Phase 5 → Deployment         (Gradio demo or REST endpoint)
```

> [!NOTE]
> Bias audit is not optional in this capstone. The rubric in Module 7 Lesson 4 requires you to run a minimum of ten adversarial prompts and document observed failure modes before a passing grade is awarded.

---

## 🔷 Technical Deep-Dive

### Step 1 — Browse and Preview a Dataset Programmatically

The code block below runs in under two minutes on any machine. It verifies that your chosen dataset is accessible, inspects its license field, and previews three rows — all before you commit to fine-tuning on it.

```python
"""
capstone/dataset_scout.py

Verifies dataset accessibility, license, and structure.
Run this before committing to a dataset for your capstone.

Requirements:
    pip install datasets huggingface_hub
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from datasets import load_dataset, DatasetDict
from huggingface_hub import DatasetCard


@dataclass
class DatasetProfile:
    name: str
    license: str
    splits: list[str]
    column_names: list[str]
    preview_rows: int = 3


APPROVED_LICENSES = {"apache-2.0", "mit", "cc-by-4.0", "cc0-1.0"}


def fetch_license(dataset_name: str) -> str:
    """Return the license string from the dataset card, or 'unknown'."""
    try:
        card = DatasetCard.load(dataset_name)
        card_data = card.data.to_dict()
        raw_license = card_data.get("license", "unknown")
        # License field may be a list or a string
        if isinstance(raw_license, list):
            return raw_license[0] if raw_license else "unknown"
        return str(raw_license)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Could not fetch dataset card: {exc}", file=sys.stderr)
        return "unknown"


def audit_license(license_str: str) -> None:
    """Warn if the license is outside the approved set."""
    normalized = license_str.lower()
    if normalized not in APPROVED_LICENSES:
        print(
            f"\n⚠️  License '{license_str}' is NOT in the approved set: "
            f"{APPROVED_LICENSES}\n"
            "    Review the dataset card before using this dataset in deployment.",
            file=sys.stderr,
        )
    else:
        print(f"✅  License '{license_str}' is approved for educational and commercial use.")


def profile_dataset(dataset_name: str, split: str = "train") -> DatasetProfile:
    """Load a dataset, print a preview, and return a DatasetProfile."""
    print(f"\n--- Profiling: {dataset_name} ---")

    license_str = fetch_license(dataset_name)
    audit_license(license_str)

    dataset: DatasetDict = load_dataset(dataset_name)
    available_splits = list(dataset.keys())

    target_split = split if split in dataset else available_splits[0]
    sample = dataset[target_split].select(range(3))

    print(f"\nAvailable splits : {available_splits}")
    print(f"Columns          : {sample.column_names}")
    print(f"\nFirst 3 rows of '{target_split}':")
    for idx, row in enumerate(sample):
        print(f"  [{idx}] {row}")

    return DatasetProfile(
        name=dataset_name,
        license=license_str,
        splits=available_splits,
        column_names=sample.column_names,
    )


if __name__ == "__main__":
    # Replace with your chosen dataset identifier from the HF Hub.
    # Examples shown for the three most common capstone task types.
    CANDIDATE_DATASETS = [
        "dair-ai/emotion",           # text classification, cc-by-4.0
        "rajpurkar/squad_v2",        # extractive Q&A, cc-by-sa-4.0 (verify)
        "cnn_dailymail",             # summarization, apache-2.0
    ]

    chosen_dataset = CANDIDATE_DATASETS[0]  # swap to your target
    profile = profile_dataset(chosen_dataset)

    print("\n--- DatasetProfile Summary ---")
    print(f"  Name    : {profile.name}")
    print(f"  License : {profile.license}")
    print(f"  Splits  : {profile.splits}")
    print(f"  Columns : {profile.column_names}")
```

**Expected output (for `dair-ai/emotion`):**

```
--- Profiling: dair-ai/emotion ---
✅  License 'cc-by-4.0' is approved for educational and commercial use.

Available splits : ['train', 'validation', 'test']
Columns          : ['text', 'label']

First 3 rows of 'train':
  [0] {'text': 'i didnt feel humiliated', 'label': 0}
  [1] {'text': 'i can go from feeling so hopeless ...', 'label': 0}
  [2] {'text': 'im grabbing a 5 day series ...', 'label': 3}

--- DatasetProfile Summary ---
  Name    : dair-ai/emotion
  License : cc-by-4.0
  Splits  : ['train', 'validation', 'test']
  Columns : ['text', 'label']
```

---

### Step 2 — Generate Your Project Plan File

Run the snippet below once to scaffold your capstone plan. Fill in the `TODO` strings before moving to Module 7 Lesson 2.

```python
"""
capstone/generate_project_plan.py

Scaffolds a project_plan.md for your capstone.
Edit the CONFIG dict, then run this script once.
"""

from __future__ import annotations

import textwrap
from pathlib import Path


CONFIG = {
    "learner_name": "TODO: your name",
    "domain": "TODO: e.g. customer support, medical notes, developer tooling",
    "task_type": "TODO: classification | qa | summarization | code_completion",
    "dataset_id": "TODO: e.g. dair-ai/emotion",
    "dataset_license": "TODO: e.g. cc-by-4.0",
    "checkpoint": "TODO: e.g. HuggingFaceTB/SmolLM2-135M",
    "compute_tier": "TODO: cpu | 8gb_vram | 16gb_vram",
    "primary_metric": "TODO: e.g. F1-macro | ROUGE-L | pass@1",
    "target_metric_value": "TODO: e.g. F1 ≥ 0.72",
}

PLAN_TEMPLATE = textwrap.dedent("""
    # Capstone Project Plan
    **Learner:** {learner_name}

    ## Domain & Task
    - **Domain:** {domain}
    - **Task type:** {task_type}
    - **Dataset:** `{dataset_id}` (License: {dataset_license})
    - **Base checkpoint:** `{checkpoint}`
    - **Compute tier:** {compute_tier}

    ## Success Criteria
    - Primary metric: **{primary_metric}** reaching **{target_metric_value}**
    - Bias audit: run ≥ 10 adversarial prompts; document all failure modes
    - Deployment: working Gradio demo or REST endpoint with input length guard

    ## Phase Checklist
    - [ ] Phase 1 — Data Preparation: clean, split (80/10/10), tokenize
    - [ ] Phase 2 — Fine-Tuning: LoRA / PEFT on `{checkpoint}`
    - [ ] Phase 3 — Evaluation: compute {primary_metric} on held-out test split
    - [ ] Phase 4 — Bias & Safety Audit: adversarial prompt log completed
    - [ ] Phase 5 — Deployment: endpoint live, input validation in place

    ## Notes
    _(Add any domain-specific constraints, data caveats, or risk flags here.)_
""")


def generate_plan(output_path: Path = Path("project_plan.md")) -> None:
    content = PLAN_TEMPLATE.format(**CONFIG)
    output_path.write_text(content, encoding="utf-8")
    print(f"✅  Project plan written to: {output_path.resolve()}")
    print("    Open the file and replace every 'TODO' before Lesson 2.")


if __name__ == "__main__":
    generate_plan()
```

> [!IMPORTANT]
> Do not proceed to Lesson 2 with any `TODO` strings still in `project_plan.md`. The plan is your contract with yourself — vague plans produce vague models.

---

## Hands-On Exercise

Work through all four steps. Each step has a verifiable output.

**Step 1 — Identify your compute tier.**
Open a terminal and run:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Record the output. Match it to the compute tier table in Core Concepts. Write the tier name in your `CONFIG` dict.

**Step 2 — Scout your dataset.**
Choose one dataset from the HF Hub that matches your domain. Run `dataset_scout.py` with that dataset identifier. Confirm the license is in the approved set. If it is not, pick a different dataset.

**Step 3 — Confirm column names map to your task.**
For classification, you need at least a `text` column and a `label` column (or equivalent). For Q&A, you need `context`, `question`, and `answers`. Write down the exact column names from your scout output — you will reference them in the <abbr title="The preprocessing step of converting raw text input into numerical tokens that a language model can process.">tokenization</abbr> step in Lesson 2.

**Step 4 — Generate and fill your project plan.**
Run `generate_project_plan.py`. Open `project_plan.md` and replace every `TODO`. Verify: no `TODO` remains, the license is approved, the metric target is specific (not "good accuracy"), and the compute tier matches Step 1.

**Verifiable outcome:** `project_plan.md` exists, contains zero instances of the string `TODO`, and the Phase Checklist shows all five phases unchecked (you will check them off as the capstone progresses).

---

## Concept Check

**Question 1**

You want to fine-tune a model on a CPU-only laptop. Which checkpoint is the correct choice?

* [x] `HuggingFaceTB/SmolLM2-135M`
* [ ] `HuggingFaceTB/SmolLM2-1.7B`
* [ ] `microsoft/phi-2`
* [ ] Any checkpoint works equally well on CPU

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** `HuggingFaceTB/SmolLM2-135M`

**Explanation:**
At 135 million parameters, SmolLM2-135M is the only checkpoint in the course's recommended set that trains and runs inference at a tolerable speed without a GPU. The 1.7B and 2.7B models require dedicated VRAM; running them on CPU makes fine-tuning impractically slow for a capstone timeline.

</details>

---

**Question 2**

Your dataset card shows the license `cc-by-nc-4.0`. You plan to deploy your capstone model as a public API. What should you do?

* [ ] Proceed — any Creative Commons license is safe for deployment
* [ ] Proceed — `nc` only restricts academic publishing
* [x] Choose a different dataset; `cc-by-nc-4.0` prohibits commercial use
* [ ] Email the dataset author for a waiver before the capstone deadline

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Choose a different dataset.

**Explanation:**
`nc` stands for Non-Commercial. A public API — even a free one — can be classified as commercial use depending on your jurisdiction and the specific terms. The safe path for a capstone that may be shared publicly is to restrict your dataset choices to `apache-2.0`, `mit`, `cc-by-4.0`, or `cc0-1.0`. Do not rely on informal interpretations of non-commercial clauses.

</details>

---

**Question 3 — Reflection Prompt**

Describe a real project where an SLM (under 2B parameters) would outperform a large general-purpose model. Justify your reasoning using at least one of these dimensions: latency, data privacy, or domain scope.

*(There is no single correct answer. Expand below for a framework to evaluate your response.)*

<details>
<summary>🔑 Click to Reveal Evaluation Framework</summary>

**Strong responses typically include:**

- A **specific domain** (e.g., on-device medical triage notes, factory floor anomaly classification, local code linting assistant)
- A **latency argument**: SLMs run on-device or on cheap inference hardware, eliminating round-trip API latency. This matters when responses must arrive in under 200 ms.
- A **privacy argument**: Sensitive data (patient records, proprietary code, financial transactions) cannot leave the organization's network. A locally deployed SLM processes data without external API calls.
- A **domain scope argument**: A model fine-tuned on 50,000 domain-specific examples often outperforms a general 70B model on narrow tasks — Brown et al. (2020) demonstrated that few-shot generalization improves with scale, but fine-tuned smaller models remain competitive on focused benchmarks.

**Weak responses** cite "cost" without specifying the deployment scale, or claim SLMs are "always better" without a constraint-based justification.

</details>

---

## Summary

- **Scope drives quality.** Choosing one task type and one domain before touching training code produces a measurable, deployable result. Broad scopes produce unfinished capstones.
- **License verification is non-negotiable.** Run `dataset_scout.py` and confirm the license string is in the approved set before any data processing begins. Fixing a license problem after fine-tuning wastes hours.
- **Compute tier determines your checkpoint.** CPU-only → SmolLM2-135M. 8 GB VRAM → SmolLM2-1.7B. 16 GB VRAM → phi-2. Mismatching tier to checkpoint is the most common cause of out-of-memory failures at the start of Module 7 Lesson 2.
- **Your project plan is a working document.** The five-phase checklist travels with you through every remaining lesson. Update it as you complete each phase.

---

## References & Credits

- Brown et al. (2020) *Language Models are Few-Shot Learners*. [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
- Hu et al. (2021) *<abbr title="Low-Rank Adaptation: an efficient fine-tuning method that freezes base model weights and injects small trainable adapter matrices.">LoRA</abbr>: Low-Rank Adaptation of Large Language Models*. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) — LoRA / <abbr title="Parameter-Efficient Fine-Tuning: techniques (like LoRA) that adapt pre-trained models by updating only a tiny fraction of parameters.">PEFT</abbr> is the fine-tuning method used in Phase 2.
- HuggingFaceTB. *SmolLM2 Model Family*. [https://huggingface.co/HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — *Last verified: 2025-06*
- Microsoft. *Phi-2 Model Card*. [https://huggingface.co/microsoft/phi-2](https://huggingface.co/microsoft/phi-2) — *Last verified: 2025-06*
- Saravia et al. (2018) *CARER: Contextualized Affect Representations for Emotion Recognition* — source paper for the `dair-ai/emotion` dataset. [https://aclanthology.org/D18-1404/](https://aclanthology.org/D18-1404/)
- Hugging Face. *Datasets Library Documentation*. [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets) — Apache 2.0 license.
---

## 📝 Chapter Quiz

**Question 1:** What is a defining characteristic of Small Language Models (SLMs) in relation to 01 Capstone Project Briefing And Dataset Selection?

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
