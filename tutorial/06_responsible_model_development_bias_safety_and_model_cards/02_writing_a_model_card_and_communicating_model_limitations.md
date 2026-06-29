# Writing a Model Card and Communicating Model Limitations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-seth/slm-development/blob/main/tutorial/06_responsible_model_development_bias_safety_and_model_cards/02_writing_a_model_card.ipynb)

| | |
|---|---|
| **Domain** | GenAI |
| **Module** | Responsible Model Development: Bias, Safety, and Model Cards |
| **Difficulty** | Beginner |
| **Estimated Time** | 25 minutes |
| **Prerequisites** | Basic Python programming knowledge; familiarity with training vs. <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr> concepts; completion of Module 5 (Optimization and Deployment) |

---

## Lesson Roadmap

- **Core Concepts** — Understand the model card framework, its origin, and the schema Hugging Face uses to structure documentation.
- **Technical Deep-Dive** — Write, validate, and publish a complete model card with front matter, bias audit findings, and evaluation results using the `huggingface_hub` SDK.
- **Hands-On Exercise** — Author a model card for a <abbr title="Low-Rank Adaptation: an efficient fine-tuning method that freezes base model weights and injects small trainable adapter matrices.">LoRA</abbr>-fine-tuned SmolLM2 adapter and push it to the Hub.
- **Concept Check** — Verify comprehension of card sections, disclosure requirements, and Hub mechanics.
- **Summary & References** — Key takeaways and the Mitchell et al. (2019) citation.

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Author a complete Hugging Face model card covering intended use, training data, evaluation results, and known limitations.
- Embed bias audit findings and concrete mitigation recommendations inside the card.
- Apply the model card framework as defined in Mitchell et al. (2019) *Model Cards for Model Reporting*.
- Publish the finished model card alongside your adapter weights on the Hugging Face Hub.

---

## 🟢 Core Concepts

### What a Model Card Is — and Why It Exists

A model card is a short, structured document that ships with a machine learning model. It answers three questions every downstream user needs answered before deployment:

1. **What does this model do, and for whom?**
2. **How was it evaluated, and where does it fail?**
3. **What risks should a deployer know about?**

Mitchell et al. (2019) *Model Cards for Model Reporting* formalized this idea. Their framework treats a model card as the "nutrition label" of ML — a standardized disclosure that lets users make informed decisions rather than relying on marketing copy.

> [!IMPORTANT]
> Mitchell et al. (2019) is the foundational citation for model card methodology. Reference it explicitly when your organization requires documented transparency practices.

### The Six Core Sections

The Hugging Face schema maps directly onto Mitchell et al.'s structure:

```
┌──────────────────────────────────────────┐
│  Model Card                              │
│                                          │
│  1. Model Summary (front matter YAML)    │
│  2. Intended Use                         │
│  3. Out-of-Scope Use                     │
│  4. Training Data                        │
│  5. Evaluation Results                   │
│  6. Bias, Risks, and Limitations         │
└──────────────────────────────────────────┘
```

**Front matter YAML** is machine-readable metadata at the top of `README.md`. The Hub parses it to populate search filters, license badges, and dataset tags automatically.

**Intended Use** describes the specific tasks, languages, and user groups the model targets. Keep it narrow and honest — overly broad claims invite misuse.

**Out-of-Scope Use** is equally important. Listing what the model *should not* do gives deployers a documented boundary and reduces liability.

**Evaluation Results** must cite the exact dataset, split, and metric used. A score without a test set name is meaningless.

**Bias, Risks, and Limitations** is where your bias audit lives. Every <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLM</abbr> fine-tuned on domain-specific data amplifies existing dataset skews. Disclosing them is not optional for responsible deployment.

### The Nutrition Label Analogy

Think of the front matter YAML as the ingredient list — quick, parseable facts. The prose sections are the extended nutritional context: what the numbers mean for *your specific use case*. Both parts are required for a complete label.

> [!NOTE]
> Hugging Face renders `README.md` from your model repository root as the model card. The front matter must be valid YAML enclosed in `---` fences — otherwise the Hub ignores the metadata entirely.

---

## 🔷 Technical Deep-Dive

### Environment Setup

```bash
pip install huggingface_hub>=0.23.0 transformers>=4.41.0 peft>=0.11.0
```

Authenticate once per environment. Store your token as an environment variable — never hardcode it.

```bash
export HF_TOKEN="hf_your_token_here"
huggingface-cli login --token $HF_TOKEN
```

### Step 1 — Create the Repository

```python
import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.environ["HF_USERNAME"]  # e.g., "priya-narayan"
REPO_NAME = "smollm2-135m-medical-triage-lora"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi()

# create_repo is idempotent — safe to re-run
create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    private=False,      # set True during development
    exist_ok=True,
)
print(f"Repository ready: https://huggingface.co/{REPO_ID}")
```

### Step 2 — Write the Model Card

The function below builds a complete, spec-compliant model card string. Every section maps to the Mitchell et al. (2019) framework.

```python
def build_model_card(
    base_model: str,
    repo_id: str,
    language: str,
    license_id: str,
    eval_results: dict[str, float],
    bias_findings: list[str],
    mitigations: list[str],
) -> str:
    """
    Construct a Hugging Face-compatible model card README.md string.

    Parameters
    ----------
    base_model    : HF model ID of the base checkpoint (e.g. 'HuggingFaceTB/SmolLM2-135M')
    repo_id       : Full 'username/repo-name' identifier for this adapter
    language      : BCP-47 language tag (e.g. 'en')
    license_id    : SPDX license identifier (e.g. 'apache-2.0')
    eval_results  : Mapping of metric name to float score (e.g. {'F1': 0.81})
    bias_findings : List of bias observations from the audit
    mitigations   : List of recommended mitigations
    """
    # ── Front matter ─────────────────────────────────────────────────────────
    eval_rows = "\n".join(
        f"  - type: {metric}\n    value: {score:.4f}"
        for metric, score in eval_results.items()
    )

    front_matter = f"""---
language:
  - {language}
license: {license_id}
base_model: {base_model}
tags:
  - peft
  - lora
  - smollm2
  - medical
  - text-classification
model-index:
  - name: {repo_id}
    results:
      - task:
          type: text-classification
        dataset:
          name: Medical Triage Subset (internal)
          type: custom
        metrics:
{eval_rows}
---
"""

    # ── Prose sections ────────────────────────────────────────────────────────
    bias_list = "\n".join(f"- {item}" for item in bias_findings)
    mitigation_list = "\n".join(f"- {item}" for item in mitigations)

    prose = f"""# {repo_id.split("/")[-1]}

> **Framework**: Model Cards for Model Reporting — Mitchell et al. (2019).
> This card follows the six-section disclosure structure defined in that work.

## Model Summary

A LoRA adapter fine-tuned on top of
[{base_model}](https://huggingface.co/{base_model})
for **emergency department triage text classification**.
The adapter routes free-text nurse notes into one of five acuity levels (ESI 1–5).

## Intended Use

| Attribute | Detail |
|---|---|
| **Primary use** | Triage acuity suggestion from nurse admission notes |
| **Primary users** | Clinical decision-support software teams |
| **Language** | English (`{language}`) |
| **Deployment context** | Requires clinician review before acting on model output |

## Out-of-Scope Use

- Autonomous clinical decision-making without human oversight.
- Languages other than English.
- Paediatric triage (training data contains fewer than 4 % paediatric cases).
- Any use outside the emergency medicine context.

> [!IMPORTANT]
> This model is **not a medical device** and has not been evaluated for regulatory
> compliance under FDA 21 CFR Part 11 or EU MDR 2017/745.

## Training Data

- **Source**: Internal de-identified ED admission notes, 2018–2022.
- **Size**: 42 000 training examples, 5 000 validation examples.
- **Preprocessing**: PHI removed via spaCy NER + manual review.
- **Known gaps**: Under-represents rural hospital notes; over-represents
  tertiary urban centres in the north-eastern United States.

## Evaluation Results

| Metric | Score | Test Set |
|---|---|---|
{''.join(
    f'| {m} | {s:.4f} | Internal holdout (n=3 000) |\\n'
    for m, s in eval_results.items()
)}

Evaluation was performed on a **stratified holdout split** — not the training set.
Performance on external hospital systems has not been validated.

## Bias, Risks, and Limitations

### Observed Bias

{bias_list}

### Recommended Mitigations

{mitigation_list}

### General Limitations

- Model confidence scores are not calibrated probabilities.
- Out-of-distribution inputs (e.g., handwriting OCR artifacts) degrade F1 by ~12 %.
- Fine-tuning is a soft control, not a safety guarantee — adversarial inputs
  can still elicit incorrect acuity assignments.

## How to Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

base_id = "{base_model}"
adapter_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForSequenceClassification.from_pretrained(base_id, num_labels=5)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

inputs = tokenizer("Patient presents with chest pain radiating to left arm.", return_tensors="pt")
logits = model(**inputs).logits
predicted_level = logits.argmax(dim=-1).item() + 1  # ESI 1-indexed
print(f"Predicted acuity: ESI-{{predicted_level}}")
```

## Citation

```bibtex
@misc{{mitchell2019model,
  title={{Model Cards for Model Reporting}},
  author={{Margaret Mitchell and Simone Wu and Andrew Zaldivar and
           Parker Barnes and Lucy Vasserman and Ben Hutchinson and
           Elena Spitzer and Inioluwa Deborah Raji and Timnit Gebru}},
  year={{2019}},
  eprint={{1810.03993}},
  archivePrefix={{arXiv}}
}}
```
"""
    return front_matter + prose
```

### Step 3 — Instantiate and Upload

```python
from huggingface_hub import upload_file
import tempfile
import pathlib

# Define evaluation results from your bias audit and test run
evaluation_scores = {
    "F1 (macro)": 0.8134,
    "Accuracy": 0.8391,
}

# Bias findings produced by the audit exercise in Lesson 1
audit_findings = [
    "ESI-3 F1 drops from 0.83 to 0.71 for notes authored by nurses with < 2 years experience.",
    "Female patient notes are assigned ESI-1 (critical) 6 % less often than matched male notes.",
    "Non-English phrases (transliterated) in otherwise English notes lower confidence by ~18 %.",
]

recommended_mitigations = [
    "Collect balanced training samples across nurse experience levels before next fine-tuning run.",
    "Add a post-hoc calibration layer and audit gender-disaggregated metrics quarterly.",
    "Preprocess notes with a language-detection step; route non-English inputs to a human reviewer.",
]

card_content = build_model_card(
    base_model="HuggingFaceTB/SmolLM2-135M",
    repo_id=REPO_ID,
    language="en",
    license_id="apache-2.0",
    eval_results=evaluation_scores,
    bias_findings=audit_findings,
    mitigations=recommended_mitigations,
)

# Write to a temp file, then upload — avoids local path assumptions
with tempfile.TemporaryDirectory() as tmpdir:
    card_path = pathlib.Path(tmpdir) / "README.md"
    card_path.write_text(card_content, encoding="utf-8")

    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="docs: add model card following Mitchell et al. (2019) framework",
    )

print(f"Model card published: https://huggingface.co/{REPO_ID}")
```

### Step 4 — Verify Front Matter Parsing

After upload, confirm the Hub parsed your YAML correctly:

```python
from huggingface_hub import model_info

info = model_info(REPO_ID)

print("Tags      :", info.tags)
print("License   :", info.cardData.get("license", "NOT FOUND"))
print("Language  :", info.cardData.get("language", "NOT FOUND"))
print("Base model:", info.cardData.get("base_model", "NOT FOUND"))

# Assertion guard — catches malformed front matter early
assert "lora" in info.tags, "Front matter YAML was not parsed — check fencing syntax."
assert info.cardData.get("license") == "apache-2.0", "License field missing."

print("\n✅ Front matter validated successfully.")
```

> [!NOTE]
> If the assertion fails, open your `README.md` and verify the `---` fences appear on their own lines with no leading whitespace. YAML indentation errors silently invalidate the entire front matter block.

---

## Hands-On Exercise

**Goal**: Publish a model card for a LoRA adapter you trained in Module 4 or 5 — or use the stub adapter ID provided below if you have not yet pushed weights.

**Time estimate**: 8–10 minutes.

### Step-by-Step

**1. Fork the starter notebook** (or open a new Jupyter cell block):

```python
# Starter: use a public stub repo if you have no adapter yet
STUB_BASE = "HuggingFaceTB/SmolLM2-135M"
YOUR_REPO_ID = f"{os.environ['HF_USERNAME']}/my-smollm2-card-exercise"

create_repo(repo_id=YOUR_REPO_ID, repo_type="model", exist_ok=True, private=True)
```

**2. Define your own domain-specific evaluation results.** Replace the medical triage metrics with any two metrics relevant to your <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">fine-tuning</abbr> task (or use placeholder values for the exercise):

```python
my_eval_results = {
    "ROUGE-L": 0.4312,
    "Exact Match": 0.2875,
}
```

**3. Run a minimal bias audit prompt sweep** to populate at least one finding. If you have no model weights locally, document a *hypothetical* gap based on your training data composition — note it as "projected" in the card.

**4. Call `build_model_card(...)` with your values**, write the output to `README.md`, and upload via `api.upload_file(...)`.

**5. Navigate to `https://huggingface.co/<your-repo-id>`** and verify:

- [ ] The model card renders with all six sections visible.
- [ ] The **Tags** sidebar shows `lora`, `peft`, and your domain tag.
- [ ] The **License** badge appears in the header.
- [ ] At least one bias finding appears under "Observed Bias".

**Verifiable outcome**: Screenshot or share the URL of your published card. All five checklist items must pass.

> [!NOTE]
> Private repositories are visible only to you while you iterate. Flip `private=False` in `create_repo` when you are ready to share.

---

## Concept Check

**Question 1**

Which file does the Hugging Face Hub read to render a model card?

* [x] `README.md` at the repository root
* [ ] `model_card.json` in the `config/` directory
* [ ] `METADATA.yaml` at the repository root
* [ ] `card.md` in the adapter directory

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** `README.md` at the repository root.

**Explanation:**
The Hub treats the root `README.md` as the model card by convention. The YAML front matter block (enclosed in `---` fences) within that file supplies machine-readable metadata. No other filename is recognized for this purpose.

</details>

---

**Question 2**

A colleague asks why you listed paediatric triage as an out-of-scope use when the model technically runs on any text input. What is the correct justification?

* [ ] The model's tokenizer rejects paediatric terminology.
* [ ] Out-of-scope sections only apply to multilingual models.
* [x] The training data under-represents paediatric cases, so performance in that sub-population is unvalidated and potentially harmful.
* [ ] <abbr title="Parameter-Efficient Fine-Tuning: techniques (like LoRA) that adapt pre-trained models by updating only a tiny fraction of parameters.">PEFT</abbr> adapters cannot generalize to unseen domains.

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C.

**Explanation:**
"Out-of-scope" does not mean technically impossible — it means the model was not validated for that population or task. Disclosing sub-population gaps follows the Mitchell et al. (2019) framework directly. Listing an unvalidated use as in-scope would misrepresent the model's reliability.

</details>

---

**Question 3 — Reflection Prompt**

> Consider a fine-tuned SLM you would build for a domain you know well (customer support, legal summarization, code review, etc.). Write two sentences describing one bias risk specific to that domain's training data and one mitigation you would document in the model card.

There is no single correct answer. Use the expandable block below to check your reasoning against the key criteria.

<details>
<summary>🔑 Click to Reveal Evaluation Criteria</summary>

A strong response will:

1. **Name a specific data skew** — not just "biased data" but *which* groups, time periods, geographies, or language registers are under-represented.
2. **Propose a concrete, actionable mitigation** — for example, targeted data collection, a post-hoc fairness metric computed per subgroup, or a human-in-the-loop review gate for flagged inputs.
3. Acknowledge that fine-tuning alone does not eliminate bias — only shifts its distribution.

If your mitigation was "retrain with more data," revisit the bias audit findings from Lesson 1 and consider disaggregated evaluation as a complementary step.

</details>

---

## Summary

- A model card is a structured disclosure document — not marketing copy. It covers intended use, training data provenance, evaluation results, and bias findings for every model you ship.
- Mitchell et al. (2019) *Model Cards for Model Reporting* defines the six-section framework that Hugging Face's `README.md` schema implements. Cite it explicitly in cards destined for organizational or regulatory review.
- The Hub front matter YAML must be enclosed in `---` fences at the top of `README.md`. Malformed YAML silently disables all metadata rendering — always validate with `model_info()` after upload.
- Bias disclosure in the card is a documentation step, not a fix. Pair it with disaggregated evaluation metrics and mitigation recommendations so downstream users can make informed deployment decisions.

---

## References & Credits

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). *Model Cards for Model Reporting*. [https://arxiv.org/abs/1810.03993](https://arxiv.org/abs/1810.03993)

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

- Hugging Face. *Model Cards Documentation* (Last verified: 2025-06). [https://huggingface.co/docs/hub/model-cards](https://huggingface.co/docs/hub/model-cards)

- Hugging Face. *SmolLM2 Model Family* (Last verified: 2025-06). [https://huggingface.co/HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)

- `huggingface_hub` Python SDK — Apache 2.0 License. [https://github.com/huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)

- `peft` library — Apache 2.0 License. [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
---

## 📝 Chapter Quiz

**Question 1:** What is a defining characteristic of Small Language Models (SLMs) in relation to 02 Writing A Model Card And Communicating Model Limitations?

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
