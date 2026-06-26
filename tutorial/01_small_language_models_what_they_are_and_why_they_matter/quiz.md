# Module 1 Quiz: Small Language Models: What They Are and Why They Matter

Test your understanding of Small Language Models: What They Are and Why They Matter.

---

## Questions

**Q1: A medical clinic wants to deploy an on-device model on a tablet that flags abnormal vital signs and surfaces relevant protocol summaries — without sending patient data to external servers. Which model choice is most appropriate, and why?**

- A) A 70B-parameter <abbr title="Large Language Model: a massive language model (7B+ parameters) requiring cloud or cluster hardware.">LLM</abbr> accessed via cloud API, because larger models produce more accurate clinical reasoning.
- B) A 1–3B-parameter <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLM</abbr> fine-tuned on clinical protocols, deployed locally on the tablet.
- C) A general-purpose SLM with no <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">fine-tuning</abbr>, because smaller models are always faster.
- D) A cloud-hosted LLM with output filtering, because on-device compute is insufficient for any NLP task.

<details>
<summary>Answer</summary>

**Correct: B**

The privacy constraint rules out any cloud API option (A and D). A fine-tuned SLM within the 1–3B range can run efficiently on tablet-class hardware and, when trained on domain-specific data, produces reliable outputs for a narrow task like protocol lookup. Option C is wrong because removing fine-tuning sacrifices the domain accuracy that clinical use requires.

</details>

---

**Q2: Which statement most accurately describes the trade-off between parameter count and task coverage in language models?**

- A) More parameters always produce lower latency because the model can cache more patterns.
- B) Fewer parameters reduce memory footprint but typically narrow the range of tasks the model handles reliably without fine-tuning.
- C) Parameter count has no effect on task coverage; training data volume is the only factor.
- D) SLMs outperform LLMs on all tasks because they overfit less to irrelevant pre-training data.

<details>
<summary>Answer</summary>

**Correct: B**

Parameter count directly affects how much world knowledge and task variety a model can encode in its weights. Smaller models use less memory and run faster, but their narrower capacity means they require targeted fine-tuning to perform reliably on specialized tasks. Option A is incorrect — more parameters increase per-token compute cost, raising latency. Option D overstates SLM capability; LLMs retain advantages on open-domain, multi-step reasoning tasks.

</details>

---

**Q3: During <abbr title="The preprocessing step of converting raw text input into numerical tokens that a language model can process.">tokenization</abbr>, the input string `"CloudWatch"` is split into the tokens `["Cloud", "Watch"]` by a BPE tokenizer. What does this tell you about the token `"CloudWatch"` in the model's vocabulary?**

- A) The tokenizer is misconfigured and should be retrained on cloud infrastructure corpora.
- B) `"CloudWatch"` appears frequently enough in the training corpus to warrant its own token, but the tokenizer chose not to include it.
- C) `"CloudWatch"` does not appear as a single merged token in the vocabulary, so the tokenizer falls back to known subword units.
- D) BPE always splits compound words regardless of vocabulary contents.

<details>
<summary>Answer</summary>

**Correct: C**

BPE builds its vocabulary by iteratively merging the most frequent adjacent byte-pair sequences found in the training corpus — Sennrich et al. (2016). If `"CloudWatch"` is absent as a single merged entry, the tokenizer decomposes it into the highest-frequency subword units it does recognize. Option B is self-contradictory: if the merged form had high enough frequency, BPE would have included it. Option D is false; BPE produces single tokens for common compound strings when they appear frequently in training data.

</details>

---

**Q4: The following Python environment setup script fails silently — the installed packages do not match the project's pinned versions. Identify the most likely root cause.**

```python
# setup_env.py
import subprocess

packages = ["torch", "transformers", "datasets", "accelerate"]

for pkg in packages:
    subprocess.run(["pip", "install", pkg])  # no version pins

print("Environment ready.")
```

- A) `subprocess.run` cannot invoke `pip`; use `os.system` instead.
- B) No version pins are specified, so `pip` installs the latest release of each package, which may conflict with each other or with the project's tested configuration.
- C) The `packages` list must be passed as a single string, not a list.
- D) `print` should appear before the loop to confirm the environment state first.

<details>
<summary>Answer</summary>

**Correct: B**

Omitting version specifiers instructs `pip` to resolve the latest available release at install time. Unpinned installs are the most common cause of silent environment drift — a newer `transformers` release may require a different `torch` ABI than the project was tested against. Option A is wrong; `subprocess.run` with a list argument is the recommended, shell-injection-safe way to call `pip`. Option C is incorrect for `subprocess.run` with `shell=False` (the default).

</details>

---

**Q5: A language model assigns the following next-token probabilities after the prompt `"The deployment region is"`: `us-east-1` → 0.61, `eu-west-2` → 0.22, `ap-southeast-1` → 0.11, `<eos>` → 0.06. The model samples with `temperature=0.1`. What output behavior should you expect, and why?**

- A) The model samples uniformly across all four tokens because temperature rescales probabilities to be equal.
- B) The model almost certainly outputs `us-east-1`, because low temperature sharpens the distribution toward the highest-probability token.
- C) The model outputs `<eos>` first because end-of-sequence tokens are always prioritized at low temperature.
- D) Low temperature causes the model to ignore the probability distribution and pick randomly.

<details>
<summary>Answer</summary>

**Correct: B**

Temperature scales the logits before the softmax step. Values below 1.0 compress the distribution, amplifying the gap between high- and low-probability tokens. At `temperature=0.1`, the already-dominant token `us-east-1` (0.61 before rescaling) becomes overwhelmingly likely after rescaling. Options A and D describe the opposite effect — high temperature or pure random sampling. Option C is incorrect; `<eos>` carries the lowest base probability and temperature reduction makes it even less likely relative to the top token.

</details>

---

**Q6: Reflect on this: A logistics company asks you to choose between fine-tuning a 350M-parameter SLM and prompting a 72B-parameter LLM via API for a package status classification task (five output classes, ~10k labeled examples available). Walk through the factors you would weigh before recommending one approach.**

- A) Fine-tune the SLM unconditionally — smaller is always better.
- B) Use the LLM API unconditionally — more parameters guarantee higher accuracy.
- C) Evaluate latency requirements, data privacy obligations, per-query cost, <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr> hardware constraints, and acceptable accuracy floor before deciding.
- D) The number of output classes determines the choice; five classes always requires an LLM.

<details>
<summary>Answer</summary>

**Correct: C**

No single metric — parameter count, class count, or dataset size — determines the right architecture. The decision matrix includes: inference latency targets (on-device SLM wins for sub-100ms needs), data privacy (on-premise SLM avoids sending shipment records externally), cost at scale (per-token API pricing compounds at high query volume), hardware budget (a 72B model requires multi-<abbr title="Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning.">GPU</abbr> serving), and whether 10k labeled examples are sufficient for SLM fine-tuning to meet the accuracy floor. Options A and B apply single-axis reasoning to a multi-dimensional problem. Option D is incorrect; classification head count is a minor architectural consideration, not a model-size driver.

</details>

---

## Capstone Challenge

**Challenge: Environment Audit and Model Selection Memo**

You have joined a three-person team building a real-time document redaction tool for a legal firm. The tool must detect and mask personally identifiable information (PII) in uploaded contracts — running entirely on the firm's on-premise Linux servers (2× NVIDIA A10G GPUs, 24 GB <abbr title="Video Random Access Memory: high-speed memory on a GPU used to store model weights and activations during run time.">VRAM</abbr> each). No contract text may leave the premises. Target latency is under 300ms per page.

Complete both parts:

**Part 1 — Environment Setup**: Write a reproducible shell script that creates an isolated Python 3.11 virtual environment, installs pinned versions of `torch`, `transformers`, `datasets`, and `accelerate`, and validates the CUDA device is visible. The script must exit with a non-zero code and a descriptive error message if CUDA is unavailable.

**Part 2 — Model Selection Memo**: Write a 200–350 word technical memo addressed to the team lead. Justify your choice of model size range (e.g., 350M, 1.3B, 3B parameters) using at least three of the following factors: VRAM budget, latency constraint, privacy requirement, task scope, and availability of labeled data. Acknowledge one risk of your chosen approach and propose a mitigation.

---

**Evaluation Rubric:**

- **Environment correctness**: The script activates a virtual environment (not the system Python), pins all four package versions explicitly, checks `torch.cuda.is_available()`, and exits non-zero with a human-readable error when CUDA is absent. Excellent work produces a script that runs end-to-end without modification on a clean Ubuntu 22.04 + CUDA 12.1 host.

- **Reasoning depth in the memo**: Excellent work names a specific parameter range, quantifies at least one constraint numerically (e.g., "a 1.3B model in fp16 occupies ~2.6 GB VRAM, well within the 24 GB ceiling"), and connects each cited factor directly to the firm's stated requirements — not to generic SLM talking points.

- **Risk acknowledgment**: Excellent work identifies a concrete, plausible risk (e.g., recall degradation on rare PII patterns not present in the fine-tuning set) and proposes a testable mitigation (e.g., augmenting training data with synthetic PII examples and measuring F1 on a held-out adversarial set).