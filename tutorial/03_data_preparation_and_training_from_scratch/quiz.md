# Module 3 Quiz: Data Preparation and Training from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-seth/slm-development/blob/main/tutorial/03_data_preparation_and_training_from_scratch/quiz.md)


Test your understanding of Data Preparation and Training from Scratch.

---

## Questions

**Q1: You are building a domain-specific dataset for a medical triage assistant. Your raw source is a collection of de-identified clinical notes stored as plain `.txt` files in a local directory. Which `datasets` API call correctly loads them as a Hugging Face `Dataset` object?**

- A) `load_dataset("text", data_dir="./clinical_notes", split="train")`
- B) `load_dataset("csv", data_files="./clinical_notes/*.txt")`
- C) `Dataset.from_text("./clinical_notes/")`
- D) `load_dataset("./clinical_notes/", streaming=True)`

<details>
<summary>Answer</summary>

**Correct: A**

`load_dataset("text", data_dir=...)` is the correct builder for plain text files; it reads each line as a `"text"` field. Option B uses the CSV builder, which expects comma-delimited columns and will fail on prose text. Option C is not a valid `datasets` API method. Option D omits the required format identifier, so the call will raise a `ValueError` about an unresolvable dataset script.

</details>

---

**Q2: A teammate shares the following training loop fragment. The model trains without throwing an error, but validation loss never improves across epochs — it stays nearly identical to the initial value. Identify the bug.**

```python
optimizer = AdamW(model.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**batch)
            val_loss += outputs.loss.item()
```

- A) `AdamW` is the wrong optimizer for language model pre-training; `SGD` should be used instead.
- B) `optimizer.zero_grad()` is missing from the training loop, causing gradient accumulation across batches.
- C) `torch.no_grad()` prevents the validation loss from being computed correctly.
- D) The model is never switched back to `model.train()` after the first validation pass.

<details>
<summary>Answer</summary>

**Correct: B**

Without `optimizer.zero_grad()` before `loss.backward()`, gradients from every previous batch accumulate. The parameter updates become dominated by accumulated noise, so the model makes little meaningful progress — validation loss stagnates. Option A is wrong: `AdamW` is the standard choice for transformer pre-training. Option C is wrong: `torch.no_grad()` is correct and necessary for memory-efficient <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr>; it does not corrupt the loss value. Option D is wrong: `model.train()` is set at the top of the outer `for epoch` loop, so it applies at the start of each epoch.

</details>

---

**Q3: Your 60M-parameter <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLM</abbr> is trained on legal contract text. You evaluate it on a held-out contract test set and observe <abbr title="A metric measuring how well a probability model predicts a sample; lower perplexity indicates higher confidence and quality.">perplexity</abbr> = 8.4. A colleague argues this means the model "gets 8.4 words wrong per sentence." Which statement most accurately describes what perplexity measures?**

- A) Perplexity equals the average number of token prediction errors per sequence.
- B) Perplexity is the exponentiated average negative log-likelihood per token — a lower value means the model assigns higher probability to the held-out text.
- C) Perplexity measures BLEU score normalized by sequence length.
- D) Perplexity is the ratio of correct token predictions to total tokens, expressed as a percentage.

<details>
<summary>Answer</summary>

**Correct: B**

Perplexity is defined as `exp(average NLL per token)`. Intuitively, it represents how many equally likely choices the model considers at each token step — lower is better. It is not a count of errors (Option A), has no relationship to BLEU (Option C), and is not an accuracy ratio (Option D). A perplexity of 8.4 on legal text is competitive, but comparing it across different tokenizers or vocabulary sizes requires care.

</details>

---

**Q4: You fine-tune an SLM to generate one-paragraph summaries of engineering incident reports. After training, you run <abbr title="Recall-Oriented Understudy for Gisting Evaluation: metrics evaluating summary quality by comparing against human references.">ROUGE</abbr>-L evaluation. The model achieves ROUGE-L F1 = 0.31 against human-written reference summaries. A stakeholder asks whether this score is "good." Which response is most accurate?**

- A) ROUGE-L F1 of 0.31 is always poor; production summarization systems require scores above 0.70.
- B) ROUGE-L measures exact n-gram overlap, so 0.31 is uninformative without domain baselines or human judgement comparison.
- C) ROUGE-L above 0.30 guarantees the summaries are factually accurate.
- D) ROUGE-L F1 of 0.31 means 31% of the generated tokens are grammatically correct.

<details>
<summary>Answer</summary>

**Correct: B**

ROUGE-L measures the longest common subsequence overlap between generated and reference text. A score of 0.31 is only meaningful relative to a domain-specific baseline — incident report summaries use terse, technical phrasing, so reference overlap is structurally lower than in news summarization benchmarks where 0.40+ is typical. ROUGE does not measure factual accuracy (Option C) or grammaticality (Option D). The absolute threshold in Option A is domain-independent and therefore misleading.

</details>

---

**Q5: You are using the Hugging Face `Trainer` API. After the first training run, you realize your `tokenize_function` forgot to set `truncation=True`. Your sequences are longer than `model.config.max_position_embeddings`. Which consequence is most likely?**

- A) The `Trainer` silently pads all sequences to the model's maximum length with no performance impact.
- B) The `DataCollatorForLanguageModeling` raises an immediate `ValueError` before training starts.
- C) The model receives input tensors wider than its positional embedding table, causing a runtime index error or corrupted attention masks.
- D) Training completes normally because PyTorch automatically truncates tensors before the forward pass.

<details>
<summary>Answer</summary>

**Correct: C**

Transformer positional embedding tables have a fixed size equal to `max_position_embeddings`. Passing a sequence longer than this limit causes an index-out-of-range error in the embedding lookup — or, in some implementations, silently wraps positions, producing corrupted attention patterns. Option A is wrong: the data collator pads short sequences but does not truncate long ones. Option B is wrong: the collator does not inspect sequence length against the model config. Option D is wrong: PyTorch performs no automatic truncation; tensor shapes are the caller's responsibility.

</details>

---

**Q6 (Reflection): Your team has trained a 45M-parameter SLM on two years of internal customer-support tickets for a software company. Perplexity on the held-out ticket set is 11.2. ROUGE-L F1 against human agent responses is 0.28. A product manager asks: "Can we ship this for auto-reply suggestions?"**

*Describe the evaluation steps you would take before recommending a production decision. Address at least three distinct evaluation dimensions beyond the metrics already computed.*

- A) *(open-ended — no multiple-choice options apply)*

<details>
<summary>Answer</summary>

**Model Answer Guidance:**

A strong response covers at least three of the following evaluation dimensions:

**1. Human evaluation of output quality.** Perplexity and ROUGE measure statistical fit and surface overlap, not user value. A sample of 100–200 generated replies should be rated by support agents on helpfulness, tone appropriateness, and factual accuracy — metrics that automation cannot fully capture.

**2. Failure mode and adversarial probing.** Run a structured set of edge-case inputs: hostile user messages, ambiguous product version references, requests involving sensitive account data. Document observed failure modes. This step directly informs the security and safety callout required before deployment — <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">fine-tuning</abbr> alone does not guarantee safe outputs.

**3. Factual grounding and hallucination rate.** Support replies often reference specific product features or policies. Evaluate what proportion of generated replies contain statements that contradict the company's documentation. Even a 5% hallucination rate is unacceptable for customer-facing text.

**4. Latency and throughput benchmarking.** Measure p50 and p99 token-generation latency under realistic concurrency. A model that produces accurate replies in 4 seconds may be unsuitable for a live chat interface where agents expect sub-1-second suggestions.

**5. Bias and demographic fairness audit.** Review outputs across different user-query styles to check for inconsistent tone, refusal rates, or quality differences that could indicate training data skew.

A response that only restates ROUGE or perplexity improvements without addressing human evaluation or failure modes should be rated **Needs Improvement**.

</details>

---

## Capstone Challenge

**Challenge: Build and evaluate a domain-specific SLM training pipeline end-to-end.**

Select a publicly available domain corpus from Hugging Face Hub — suitable options include `scientific_papers` (arXiv split), `eurlex` (EU legal text), or a filtered subset of `pile-of-law`. Complete all three stages below and document your results in a structured evaluation report.

**Stage 1 — Dataset Construction:** Load the corpus using `load_dataset`, apply a `map`-based <abbr title="The preprocessing step of converting raw text input into numerical tokens that a language model can process.">tokenization</abbr> function with truncation and padding to a fixed block length, filter out sequences below 64 tokens, and produce a 90/5/5 train/validation/test split. Log the final token count per split.

**Stage 2 — Training:** Train a `GPT-2`-architecture model (config: 6 layers, 8 heads, 512 hidden dim — approximately 45M parameters) for at least 3 epochs using either the `Trainer` API or a manual PyTorch loop. Use `AdamW` with a linear warmup schedule. Save a checkpoint after each epoch.

**Stage 3 — Evaluation:** Compute perplexity on the test split. If your domain involves generation (e.g., summarization or completion), compute ROUGE-L F1 against at least 50 reference outputs. Run a minimum of 10 adversarial or edge-case prompts and document observed failure modes.

---

**Evaluation Rubric:**

- **Dataset pipeline correctness:** Tokenization applies truncation and padding; no raw sequences exceed `max_length`; split sizes are logged and plausible for the source corpus; filtering logic removes degenerate sequences without discarding more than 15% of the total data.

- **Training loop integrity:** <abbr title="The algorithm (e.g. AdamW) that updates model weights based on computed gradients to minimize the loss.">Optimizer</abbr>, scheduler, and gradient zeroing are all correctly implemented; validation loss is logged per epoch and shows a downward trend; at least one checkpoint is saved and reloadable with `model.from_pretrained()`.

- **Evaluation completeness:** Perplexity is computed on the held-out test split (not the validation split); ROUGE-L scores are reported with the reference source documented; the adversarial prompt log names at least three distinct failure categories (e.g., hallucination, topic drift, unsafe content) with representative examples.

- **Analytical commentary:** The evaluation report interprets metrics relative to a stated domain baseline or prior published result — it does not report a ROUGE score without context. The report includes at least one concrete recommendation for what additional data or training change would most likely improve the weakest metric.

| Grade | Description |
|---|---|
| **Excellent (Pass)** | All three stages complete; metrics computed on correct splits; adversarial log present; commentary references a domain baseline. |
| **Satisfactory** | All three stages attempted; minor methodological gap (e.g., perplexity computed on validation set rather than test set); no adversarial log but failure modes briefly discussed. |
| **Needs Improvement** | One or more stages incomplete; metrics absent or computed on training data; no qualitative analysis of outputs. |