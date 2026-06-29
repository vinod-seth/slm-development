# Module 2 Quiz: Transformer Architecture for Practitioners

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-seth/slm-development/blob/main/tutorial/02_transformer_architecture_for_practitioners/quiz.md)


Test your understanding of Transformer Architecture for Practitioners.

---

## Questions

**Q1: A teammate claims that removing residual connections from a 12-layer transformer "shouldn't matter much since the attention layers do the real work." What is the most accurate rebuttal?**

- A) Residual connections are only needed in models with more than 1B parameters, so the claim may be correct for <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLMs</abbr>.
- B) Residual connections allow gradients to flow directly to early layers during <abbr title="The algorithm that calculates gradients of the loss function with respect to weights by moving backward through the network.">backpropagation</abbr>, preventing vanishing gradients that would make deep stacking untrainable.
- C) Residual connections reduce memory usage by sharing weights across layers, so removing them would increase <abbr title="Video Random Access Memory: high-speed memory on a GPU used to store model weights and activations during run time.">VRAM</abbr> consumption.
- D) Residual connections perform layer normalization, which stabilizes training — without them, activations would explode.

<details>
<summary>Answer</summary>

**Correct: B**

Residual connections create a direct path from the output of each sub-layer back to its input, which preserves gradient signal across many layers during backpropagation. Without them, gradients diminish exponentially through each layer, making training a 12-layer network effectively impossible. Option D confuses residual connections with layer normalization, which is a separate operation; Option C describes weight tying, not residuals; Option A is incorrect — residuals are critical at every scale.

</details>

---

**Q2: You are planning to fine-tune a 360M-parameter SLM on a single consumer <abbr title="Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning.">GPU</abbr> with 8 GB of VRAM. The model uses fp32 weights by default. Which statement most accurately describes your situation?**

- A) Training in fp32 is fine — 360M parameters at 4 bytes each occupies roughly 1.4 GB, well within the 8 GB budget.
- B) The raw weights fit in VRAM, but optimizer states and activations during a forward/backward pass will push total usage well beyond 8 GB in a standard training loop.
- C) You must upgrade to a 24 GB GPU immediately; no training configuration can fit a 360M model on 8 GB.
- D) VRAM is only consumed by the model weights, so 8 GB is more than sufficient for any batch size.

<details>
<summary>Answer</summary>

**Correct: B**

The weights alone at fp32 occupy approximately 1.44 GB (360M × 4 bytes), but a standard training loop also holds optimizer states (Adam stores two momentum terms per parameter, tripling weight memory), gradients, and intermediate activations. Combined, these can easily consume 10–18 GB for even a modest batch size. Option A ignores everything outside the weight tensor. Option C overstates the constraint — techniques like gradient checkpointing and bf16 mixed precision make 8 GB viable. Option D is incorrect; activations and optimizer states are significant VRAM consumers.

</details>

---

**Q3: Examine the following code snippet intended to inspect the attention configuration of a <abbr title="A model trained on a massive general dataset to learn language patterns before fine-tuning.">pretrained</abbr> SLM. Identify the bug.**

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-360M")
cfg = model.config

print(f"Heads: {cfg.num_attention_heads}")
print(f"Head dimension: {cfg.hidden_size // cfg.num_key_value_heads}")
print(f"Layers: {cfg.num_hidden_layers}")
```

- A) `AutoModel` should be replaced with `AutoModelForCausalLM` to access the config object.
- B) The head dimension formula divides `hidden_size` by `num_key_value_heads` instead of `num_attention_heads`, producing a misleading value when grouped-query attention (GQA) is used.
- C) `from_pretrained` requires an explicit `cache_dir` argument or it will raise a `FileNotFoundError`.
- D) `model.config` is not accessible directly; the config must be loaded separately via `AutoConfig.from_pretrained`.

<details>
<summary>Answer</summary>

**Correct: B**

Head dimension is defined as `hidden_size // num_attention_heads`. Dividing by `num_key_value_heads` gives the correct denominator only when the model uses multi-head attention with equal query and key-value heads, but yields an inflated, incorrect value for grouped-query attention (GQA) models — which SmolLM2-360M uses. Option A is wrong; `AutoModel` does expose `.config`. Option C is wrong; `from_pretrained` caches to `~/.cache/huggingface` by default without error. Option D is wrong; `model.config` is a standard attribute on all `PreTrainedModel` instances.

</details>

---

**Q4: In a transformer's feed-forward network (FFN) sub-layer, the intermediate dimension is typically 4× the model's hidden dimension. A colleague proposes halving this multiplier to 2× across all layers of a 135M-parameter SLM to reduce <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr> latency. What is the most likely trade-off?**

- A) Halving the FFN width has no effect on latency because attention, not the FFN, dominates compute at this scale.
- B) Halving the FFN width reduces parameter count and compute per token, but risks degrading the model's ability to store and retrieve factual associations, which research links to FFN capacity.
- C) The FFN intermediate dimension must always equal 4× hidden size due to a hard constraint in the `transformers` library; the change would raise a `ValueError` at load time.
- D) Reducing the FFN width increases VRAM consumption because smaller tensors require padding to align with GPU memory boundaries.

<details>
<summary>Answer</summary>

**Correct: B**

The FFN sub-layer is widely understood to function as a key-value memory store for factual knowledge (Geva et al., 2021). Reducing its capacity lowers parameter count and arithmetic intensity per token — which does improve throughput — but compresses the space available for storing learned associations, often degrading factual recall and task accuracy. Option A is incorrect; at sequence lengths typical of SLM inference, FFN layers contribute substantially to total FLOPs. Option C is false; the multiplier is a configurable hyperparameter. Option D reverses the relationship between tensor size and VRAM.

</details>

---

**Q5: You run the following snippet to verify tokenizer–model alignment before <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">fine-tuning</abbr> and observe an unexpected output. What is the most likely root cause?**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")

sample = "Transformer blocks stack attention and FFN sub-layers."
ids = tokenizer(sample, return_tensors="pt")
output = model(**ids)
print(output.logits.shape)
# Expected: [1, seq_len, vocab_size]
# Actual:   RuntimeError — size mismatch on embedding weight
```

- A) `return_tensors="pt"` is invalid; the tokenizer requires `return_tensors="tf"` by default.
- B) The tokenizer from the 135M checkpoint and the model from the 360M checkpoint may have different vocabulary sizes, causing an embedding dimension mismatch at the model's input layer.
- C) `model(**ids)` is incorrect syntax; you must unpack `input_ids` and `attention_mask` as separate keyword arguments.
- D) The `logits` attribute is only available after calling `model.eval()`; training mode suppresses it.

<details>
<summary>Answer</summary>

**Correct: B**

Different checkpoint variants of the same model family can carry different vocabulary sizes — for example, if the 135M and 360M variants were trained with distinct tokenizers or vocabulary expansions. The embedding matrix is sized `[vocab_size, hidden_size]`, so a mismatch between tokenizer vocabulary size and the model's embedding table triggers a dimension error immediately. Always load the tokenizer and model from the same checkpoint path. Option C is wrong; `model(**ids)` correctly unpacks the tokenizer's output dict. Option D is incorrect; `logits` are available in both training and eval modes. Option A is wrong; `"pt"` is the standard PyTorch tensor format.

</details>

---

**Q6: A team running on a single RTX 4090 (24 GB VRAM) wants to train a 1.7B-parameter SLM from scratch using bf16 precision with the AdamW optimizer. Without gradient checkpointing or any memory optimization, will this fit? Select the most accurate analysis.**

- A) Yes, comfortably. The weights at bf16 occupy 3.4 GB, leaving 20.6 GB for everything else.
- B) No. Weights alone at bf16 occupy 3.4 GB, but AdamW adds two fp32 momentum buffers (~13.6 GB), gradients add ~3.4 GB, and activations at a reasonable batch size push total usage to 25–35 GB — exceeding 24 GB.
- C) Yes, because the RTX 4090 automatically offloads optimizer states to system RAM without configuration.
- D) No, but only because bf16 is not supported on consumer GPUs; switching to fp16 would make it fit.

<details>
<summary>Answer</summary>

**Correct: B**

At bf16, 1.7B parameters consume approximately 3.4 GB. AdamW maintains two fp32 momentum tensors per parameter, adding roughly 13.6 GB. <abbr title="A vector of partial derivatives indicating how to adjust model weights to minimize the loss function.">Gradients</abbr> at bf16 add another 3.4 GB. Activations — even with a batch size of 1 and moderate sequence length — typically add several more gigabytes, placing realistic peak usage well above 24 GB. Option A accounts only for weight storage and ignores optimizer and gradient memory entirely. Option C is incorrect; automatic <abbr title="Central Processing Unit: the general-purpose processor in a computer.">CPU</abbr> offloading requires explicit configuration via libraries like DeepSpeed or Accelerate. Option D is wrong; the RTX 4090 supports both bf16 and fp16 natively.

</details>

---

## Capstone Challenge

**Architectural Audit and Hardware Feasibility Report**

Select any publicly available SLM on Hugging Face with fewer than 500M parameters (for example, `HuggingFaceTB/SmolLM2-360M`, `microsoft/phi-2` at 2.7B is excluded — stay under the limit). Write a structured report covering all three areas below.

**Part 1 — Architectural Inventory**: Load the model's config and extract the following values programmatically: number of hidden layers, hidden dimension, number of query attention heads, number of key-value heads (to identify GQA vs. MHA), FFN intermediate dimension, vocabulary size, and maximum sequence length. Present these in a formatted table and calculate the theoretical parameter count from the config values. Compare your calculation against the reported count in `model.num_parameters()`.

**Part 2 — Memory Footprint Projection**: Compute the expected VRAM requirement for inference (weights only) and for a full fine-tuning run (weights + AdamW states + gradients + activations at batch size 4, sequence length 512) in both fp32 and bf16. Produce a hardware-tier recommendation table mapping each scenario to a specific consumer or cloud GPU tier.

**Part 3 — <abbr title="A mechanism that lets neural networks focus on specific parts of the input sequence when generating output.">Attention</abbr> Pattern Inspection**: Run a forward pass on a five-sentence input of your choice with `output_attentions=True`. Extract the attention weight tensor from the final layer and describe — in two to three paragraphs — what the attention distribution across heads reveals. Identify any heads that appear to attend broadly versus narrowly, and comment on what this might suggest about specialization.

---

**Evaluation Rubric:**

- **Config extraction correctness**: All eight config fields are extracted programmatically (not manually copied from the model card), the theoretical parameter calculation matches `num_parameters()` within 1%, and the table is clearly formatted with correct units.
- **Memory projection accuracy**: Fp32 and bf16 projections correctly account for weights, AdamW dual momentum buffers, gradients, and activation estimates; the hardware-tier table maps each scenario to a real, named GPU with documented VRAM; recommendations are justified with explicit arithmetic.
- **Attention analysis depth**: The forward pass uses `output_attentions=True` correctly, tensor shapes are verified before interpretation, and the written analysis distinguishes between broad and narrow attention patterns with specific reference to head indices and layer position — avoiding vague generalizations like "some heads attend to different tokens."
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
