# Module 4 Quiz: Fine-Tuning Pretrained SLMs with PEFT and LoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-seth/slm-development/blob/main/tutorial/04_fine_tuning_pretrained_slms_with_peft_and_lora/quiz.md)


Test your understanding of <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">Fine-Tuning</abbr> Pretrained <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLMs</abbr> with <abbr title="Parameter-Efficient Fine-Tuning: techniques (like LoRA) that adapt pre-trained models by updating only a tiny fraction of parameters.">PEFT</abbr> and <abbr title="Low-Rank Adaptation: an efficient fine-tuning method that freezes base model weights and injects small trainable adapter matrices.">LoRA</abbr>.

---

## Questions

**Q1: A team wants to fine-tune a 360M-parameter SLM for legal document summarization. They have 8 GB of <abbr title="Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning.">GPU</abbr> <abbr title="Video Random Access Memory: high-speed memory on a GPU used to store model weights and activations during run time.">VRAM</abbr> and 4,000 labeled examples. Which training approach is most appropriate?**

- A) Full fine-tuning with all parameters unfrozen and a learning rate of 5e-4
- B) Training from scratch on the 4,000 examples using a randomly initialized model
- C) LoRA fine-tuning targeting the attention projection matrices with rank 8
- D) Freezing all layers and training only a task-specific classification head

<details>
<summary>Answer</summary>

**Correct: C**

LoRA fine-tuning injects trainable low-rank matrices into selected layers while keeping the base weights frozen. This keeps VRAM consumption well within the 8 GB budget and extracts the most value from <abbr title="A model trained on a massive general dataset to learn language patterns before fine-tuning.">pretrained</abbr> weights when labeled data is limited. Full fine-tuning (A) risks catastrophic forgetting and exceeds typical VRAM limits at this scale. Training from scratch (B) requires orders of magnitude more data. A frozen backbone with only a classification head (D) is better suited to classification tasks, not generative summarization.

</details>

---

**Q2: The following LoRA configuration is throwing a `ValueError` at training time. Identify the problem.**

```python
from peft import LoraConfig, TaskType

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=0,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
```

- A) `lora_alpha` must equal `r` for stable training
- B) `r=0` is invalid; LoRA rank must be a positive integer
- C) `target_modules` cannot include both `q_proj` and `v_proj` simultaneously
- D) `bias="none"` conflicts with `TaskType.CAUSAL_LM`

<details>
<summary>Answer</summary>

**Correct: B**

The rank `r` controls the dimensionality of the low-rank decomposition matrices. Setting `r=0` means no learnable parameters exist, which is mathematically undefined and causes a `ValueError` in the PEFT library. A common starting value is `r=8` or `r=16`. The other options are distractors: `lora_alpha` and `r` need not be equal (alpha acts as a scaling factor), targeting multiple projection matrices is standard practice, and `bias="none"` is a valid and common setting.

</details>

---

**Q3: In the context of transfer learning, what does "catastrophic forgetting" mean, and which PEFT mechanism directly mitigates it?**

- A) The model forgets the fine-tuning data when the learning rate is too high; mitigated by gradient clipping
- B) The model's pretrained representations degrade when all weights are updated on a small task-specific dataset; mitigated by freezing base weights and training only adapter parameters
- C) The optimizer forgets momentum statistics when training is interrupted; mitigated by saving optimizer checkpoints
- D) The tokenizer vocabulary becomes misaligned with new domain text; mitigated by vocabulary extension

<details>
<summary>Answer</summary>

**Correct: B**

Catastrophic forgetting occurs when gradient updates for a narrow task overwrite the broadly useful representations learned during pretraining. PEFT methods like LoRA sidestep this by keeping base weights frozen and routing task-specific learning through small adapter matrices. This preserves general language understanding while adding task specialization. Options A, C, and D describe real training concerns — learning rate instability, checkpoint management, and domain vocabulary mismatch — but none of them define catastrophic forgetting.

</details>

---

**Q4: You are building an instruction-following dataset for a medical triage SLM. Which prompt template structure best supports instruction fine-tuning?**

- A) Concatenating raw Q&A pairs separated by newlines with no structural markers
- B) A structured format with explicit `### Instruction:`, `### Context:`, and `### Response:` delimiters, consistently applied across all examples
- C) Free-form prose descriptions of each task with the expected output appended after a dash
- D) A JSON object per example with keys `input` and `label`, passed directly to the tokenizer without formatting

<details>
<summary>Answer</summary>

**Correct: B**

Instruction fine-tuning trains the model to recognize and respond to a consistent signal structure. Explicit delimiters like `### Instruction:` give the model repeatable anchors that generalize at <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr> time. Without consistent structure (A, C), the model cannot reliably separate the instruction from the context or the response boundary. Raw JSON keys (D) are not text tokens the causal LM interprets meaningfully unless you format them into a template string first.

</details>

---

**Q5: A colleague claims that increasing LoRA rank `r` from 8 to 64 will always improve fine-tuning quality. Evaluate this claim.**

- A) Correct — higher rank captures more task-specific information and cannot hurt performance
- B) Incorrect — higher rank linearly increases full-model parameters, negating the PEFT advantage entirely
- C) Partially correct — higher rank increases adapter capacity and trainable parameter count, but risks <abbr title="A training error where a model learns training data details too well, performing poorly on new data.">overfitting</abbr> on small datasets and increases VRAM usage
- D) Incorrect — ranks above 16 are unsupported by the PEFT library and will raise an error

<details>
<summary>Answer</summary>

**Correct: C**

Rank is a capacity dial, not a quality guarantee. Increasing `r` expands the adapter's expressiveness, which can help when the task is complex and data is plentiful. On small datasets (fewer than ~10K examples), high-rank adapters overfit readily. VRAM cost also scales with rank since each adapter adds two matrices of dimensions `(hidden_size × r)` and `(r × hidden_size)`. The base model weights remain frozen regardless of rank, so option B overstates the parameter impact. Option D is false — the PEFT library supports arbitrary positive integer ranks.

</details>

---

**Q6: A security-focused reviewer flags your instruction-tuned SLM deployment. They note that the system prompt instructs the model to "never reveal confidential patient data." Why is this an insufficient safety guarantee, and what additional control should you add?**

- A) System prompts are sufficient; the reviewer is incorrect because instruction fine-tuning enforces hard behavioral constraints
- B) System prompts and fine-tuning are soft controls that can be overridden by adversarial inputs; complement them with output filtering, input length limits, and documented failure-mode testing
- C) The fix is to increase LoRA rank so the model better internalizes the safety instruction during training
- D) Replace the system prompt with a classification head that blocks sensitive outputs at the logit level

<details>
<summary>Answer</summary>

**Correct: B**

Instruction fine-tuning and system prompts shape model behavior probabilistically — they are not enforcement mechanisms. A sufficiently crafted adversarial prompt can bypass them. Production deployments require defense in depth: an output content filter as a post-processing gate, input length and structure validation to reduce injection surface area, and a documented set of adversarial test cases run before each release. Increasing LoRA rank (C) adds capacity but does not encode rules. A logit-level classification head (D) is a valid complementary tool but does not replace the full set of controls described in B.

</details>

---

## Capstone Challenge

You work at a small healthtech startup. Your team has a pre-trained `SmolLM2-360M` checkpoint and 6,200 annotated patient intake summaries, each paired with a structured triage category and a one-sentence clinical rationale. Your deployment target is a <abbr title="Central Processing Unit: the general-purpose processor in a computer.">CPU</abbr>-only edge server with 4 GB RAM.

**Your task:** Design and document a complete LoRA fine-tuning pipeline that produces a deployable instruction-following SLM for this triage use case. Your submission must include:

1. A `LoraConfig` with justified hyperparameter choices (`r`, `lora_alpha`, `target_modules`, `lora_dropout`)
2. A prompt template used consistently in training and at inference
3. A training script (or pseudocode with key implementation decisions annotated) using `transformers` + `peft`
4. An evaluation plan referencing at least one quantitative metric (<abbr title="Recall-Oriented Understudy for Gisting Evaluation: metrics evaluating summary quality by comparing against human references.">ROUGE</abbr>-L, F1, or per-category accuracy)
5. A brief adversarial prompt audit — run five edge-case inputs and document observed model behavior and any failure modes

**Evaluation Rubric:**

- **LoRA configuration justification**: Rank, alpha, and target module choices are explicitly linked to the memory budget and dataset size — not copied from a tutorial without reasoning. An excellent submission explains why `r=8` or `r=16` was chosen over `r=64` given 6,200 training examples.
- **Prompt template consistency**: The same delimiter structure appears in the training collator and the inference call. An excellent submission shows both the training-time formatted string and the inference-time prompt side by side, with no structural mismatch.
- **Evaluation rigor**: An excellent submission reports ROUGE-L or per-category F1 on a held-out split (at minimum 10% of the dataset), not just training loss. Bonus: include a confusion matrix across triage categories.
- **Safety and failure-mode documentation**: An excellent submission lists at least five adversarial or out-of-distribution prompts (e.g., inputs in a non-English language, inputs with no clinical content, inputs exceeding max token length), records the model's actual output, and proposes a mitigation for each observed failure.
- **Edge deployment readiness**: An excellent submission notes that the LoRA adapter will be merged into the base weights before export (`merge_and_unload()`), confirms the resulting model fits within 4 GB RAM, and specifies the <abbr title="The process of reducing weight precision (e.g. from 16-bit to 4-bit) to shrink model size and speed up inference.">quantization</abbr> strategy (e.g., 8-bit dynamic quantization via `torch.quantization`) used to meet the constraint.
---

## 📝 Chapter Quiz

**Question 1:** What is a defining characteristic of Small Language Models (SLMs) in relation to Quiz?

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
