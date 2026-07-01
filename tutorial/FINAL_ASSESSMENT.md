# Small Language Model (SLM) Development: Comprehensive Final Assessment

Welcome to the Final Assessment for Small Language Model Development! This examination evaluates your end-to-end knowledge across transformer architectures, data preparation, training loops from scratch, LoRA fine-tuning, quantization, serving, and responsible model auditing.

---

## 📋 Assessment Overview
- **Total Questions:** 25 Comprehensive Questions
- **Passing Score:** 80% (20/25 correct)
- **Coverage:** Modules 01 through 07
- **Format:** Single best answer. Each question has exactly one correct option.

---

## 🟢 Section 1: SLM Fundamentals & Transformer Architecture (Modules 01–02)

**Question 1:** Why are Small Language Models (SLMs) increasingly favored for enterprise edge deployments compared to 70B+ LLMs?
* [ ] They are easier to write in HTML
* [x] They offer sub-100ms latency, lower VRAM requirements, and superior parameter efficiency for domain-specific tasks
* [ ] They do not use attention mechanisms
* [ ] They do not require any training data

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** SLMs provide exceptional task-specific accuracy while drastically reducing hardware costs, latency, and operational footprint.
</details>

---

**Question 2:** There is no single universal parameter threshold for an SLM. How does NVIDIA's 2025 position paper practically define one?
* [ ] Any model with exactly 1 billion parameters
* [ ] Any model trained on fewer than 100 billion tokens
* [x] A model that can run on a common consumer device and respond fast enough to serve a single user
* [ ] A model that cannot perform in-context learning

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C

**Explanation:** A deployment-relative definition survives hardware advances: as consumer devices get faster, the practical "SLM ceiling" rises, unlike a fixed parameter count.
</details>

---

**Question 3:** Within a single transformer block, which component typically holds the **majority** of the non-embedding parameters?
* [ ] The layer normalization modules
* [ ] The positional encodings
* [x] The position-wise feed-forward network (FFN/MLP)
* [ ] The softmax function

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C

**Explanation:** The FFN expands the hidden dimension (commonly ~4x) and contracts it back, holding roughly two-thirds of a transformer's non-embedding parameters — which is where much stored knowledge resides.
</details>

---

**Question 4:** What is the primary benefit of Grouped-Query Attention (GQA), widely adopted in modern SLMs?
* [ ] It removes the need for positional encodings
* [x] It shrinks the KV cache by sharing key/value heads across query heads, with near-MHA quality
* [ ] It makes attention computation linear in sequence length
* [ ] It eliminates the feed-forward network

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** GQA groups query heads to share a few KV heads, cutting KV-cache memory (critical at long context and on constrained hardware) while retaining most of the quality of full multi-head attention. Note it reduces the KV cache, not the quadratic attention compute.
</details>

---

**Question 5:** Why do most modern SLMs use Rotary Position Embeddings (RoPE) instead of learned absolute positional embeddings?
* [ ] RoPE adds a large number of trainable parameters that improve capacity
* [x] RoPE encodes relative position, adds no parameters, and extrapolates better to longer contexts
* [ ] RoPE removes the need for attention masks
* [ ] RoPE only works for encoder-only models

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** RoPE rotates query/key vectors by an angle proportional to position, so attention depends on relative distance. It is parameter-free and generalizes to longer sequences, enabling context-scaling techniques.
</details>

---

**Question 6:** How do the compute and memory of standard self-attention scale with sequence length *n*?
* [ ] Linearly, O(n)
* [ ] Logarithmically, O(log n)
* [x] Quadratically, O(n²)
* [ ] Constant, O(1)

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C

**Explanation:** Attention compares every token with every other token, producing an n×n score matrix. This quadratic cost caps affordable context and motivates FlashAttention, sliding-window, and sparse attention.
</details>

---

## 🔵 Section 2: Data Preparation, Training & Evaluation (Module 03)

**Question 7:** Why is deduplication of training data important for language-model quality?
* [ ] It increases the total token count
* [x] It reduces memorization, improves generalization, and helps prevent test-set leakage
* [ ] It is only needed for images, not text
* [ ] It speeds up tokenizer training but harms the model

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** Duplicates cause over-memorization, waste compute, and can leak evaluation content into training. Removing them improves data efficiency and generalization — especially important for small-capacity models.
</details>

---

**Question 8:** In causal language-model pretraining, what does "packing" (concatenate-then-chunk) accomplish?
* [ ] It pads every document to a global maximum length
* [x] It concatenates tokenized documents and splits them into fixed-length blocks, eliminating padding waste
* [ ] It removes all special tokens from the corpus
* [ ] It converts the dataset into images

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** Packing keeps every position useful by filling fixed context-length blocks from a concatenated token stream, maximizing tokens-per-step instead of wasting positions on padding.
</details>

---

**Question 9:** What does **perplexity** measure, and what does a lower value indicate?
* [ ] The number of parameters in the model; lower is smaller
* [x] The exponential of average per-token cross-entropy; lower means the model predicts held-out text better
* [ ] The GPU memory used; lower is cheaper
* [ ] The number of attention heads; lower is faster

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** Perplexity reflects how "surprised" a model is by real text. It is comparable only across the same tokenizer and data, and low perplexity does **not** guarantee good task performance, so it must be paired with task evaluation.
</details>

---

**Question 10:** During instruction fine-tuning, why are the prompt/instruction tokens typically masked from the loss (labels set to -100)?
* [ ] To make training run faster on CPU
* [ ] To encrypt the prompt
* [x] So the loss is computed only on the assistant's response, teaching generation rather than prompt prediction
* [ ] To increase the vocabulary size

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C

**Explanation:** Completion-only loss focuses the model's capacity on producing good responses instead of predicting the user's words, which sharpens instruction following.
</details>

---

**Question 11:** For mixed-precision AdamW training, the common estimate is roughly **16 bytes per parameter**. Which breakdown explains it?
* [ ] 16 bytes of FP16 weights only
* [x] 2 (FP16 weights) + 2 (FP16 gradients) + 4 (FP32 master weights) + 8 (FP32 Adam moments)
* [ ] 8 bytes weights + 8 bytes activations
* [ ] 4 bytes each for weights, gradients, activations, and vocabulary

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** Optimizer state dominates: AdamW keeps two FP32 moment buffers per parameter (8 bytes) plus an FP32 master copy (4 bytes) on top of FP16 weights and gradients (2+2). Activations are additional.
</details>

---

## 🟣 Section 3: Fine-Tuning with PEFT & LoRA (Module 04)

**Question 12:** How does LoRA reduce the number of trainable parameters during fine-tuning?
* [ ] It deletes most layers of the model
* [x] It freezes the base weights and trains small low-rank update matrices (B·A) instead
* [ ] It converts the model to 1-bit precision
* [ ] It trains only the tokenizer

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** With the base frozen, no gradients or optimizer states are stored for it; only the tiny adapters are trained. Since optimizer state is the largest training cost, this collapses memory usage.
</details>

---

**Question 13:** In **QLoRA**, in what format is the frozen base model loaded?
* [ ] FP32
* [ ] INT8 symmetric
* [x] 4-bit NormalFloat (NF4)
* [ ] Binary (1-bit)

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C

**Explanation:** QLoRA loads the base in 4-bit NF4 (a data type suited to normally distributed weights), adds double quantization and paged optimizers, and trains 16-bit LoRA adapters on top — enabling large-model fine-tuning on a single consumer GPU.
</details>

---

**Question 14:** What is the recommended best practice for choosing LoRA **target modules** to maximize quality?
* [ ] Apply LoRA only to the embedding layer
* [ ] Apply LoRA only to `q_proj` and `v_proj`
* [x] Apply LoRA to all linear layers (attention **and** MLP projections)
* [ ] Never apply LoRA to attention layers

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C

**Explanation:** Targeting all linear layers — including the MLP/FFN projections, not just query/value — gives the adapter more places to influence the model and typically closes the gap with full fine-tuning.
</details>

---

**Question 15:** What is **catastrophic forgetting**, and why does LoRA inherently reduce it compared to full fine-tuning?
* [ ] It is a GPU memory error; LoRA adds more VRAM
* [x] It is the loss of previously learned abilities; LoRA keeps the base weights frozen, structurally preserving prior knowledge
* [ ] It is a tokenizer mismatch; LoRA retrains the tokenizer
* [ ] It is overfitting to the test set; LoRA removes the test set

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** Because LoRA trains only added adapters over a frozen base, the original knowledge cannot be overwritten and adapters can even be disabled to recover base behavior — unlike full fine-tuning, which mutates weights in place.
</details>

---

**Question 16:** In a LoRA configuration, how is the effective scaling of the low-rank update computed?
* [ ] r × alpha
* [x] alpha / r
* [ ] alpha + r
* [ ] r − alpha

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** The adapter output is scaled by `alpha / r`, so `alpha` and `r` interact and must be tuned together; raising `r` without adjusting `alpha` would otherwise change the effective update magnitude.
</details>

---

## 🟠 Section 4: Optimization & Deployment (Module 05)

**Question 17:** Approximately how much smaller is a model's weight footprint at **4-bit** quantization compared to **FP16**?
* [ ] About the same size
* [ ] About 2× smaller
* [x] About 4× smaller
* [ ] About 16× smaller

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C

**Explanation:** FP16 uses 2 bytes per parameter; 4-bit uses about 0.5 bytes — roughly a 4× reduction (before KV-cache and overhead), which is what lets SLMs fit edge devices.
</details>

---

**Question 18:** What problem does **PagedAttention** (popularized by vLLM) primarily solve?
* [ ] It compresses model weights to 2-bit
* [x] It eliminates KV-cache memory fragmentation, allowing far higher serving concurrency and throughput
* [ ] It removes the need for a tokenizer
* [ ] It trains the model faster

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** PagedAttention stores the KV cache in non-contiguous pages (like OS virtual memory), removing fragmentation so many more sequences share GPU memory — a key enabler of high-throughput serving.
</details>

---

**Question 19:** What distinguishes **continuous (in-flight) batching** from static batching?
* [ ] It uses a single fixed, very large batch for the whole day
* [x] It admits new requests and evicts finished ones every token iteration, keeping the GPU saturated
* [ ] It processes exactly one request at a time
* [ ] It only works on CPUs

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** Static batching waits for the longest sequence in a fixed batch, idling the GPU as it drains. Continuous batching schedules at the token level, dramatically raising throughput under mixed-length, bursty traffic.
</details>

---

**Question 20:** Activation quantization is difficult mainly because of a few extreme-magnitude "outlier" features. Which technique addresses this by migrating the difficulty onto the weights?
* [ ] Beam search
* [ ] Gradient checkpointing
* [x] SmoothQuant
* [ ] Rotary embeddings

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C

**Explanation:** SmoothQuant applies a per-channel scaling that divides activations and multiplies the corresponding weights (a mathematically equivalent transform), shifting the dynamic-range burden onto the robust weights so both can be quantized to low precision.
</details>

---

**Question 21:** In LLM serving, **Time-To-First-Token (TTFT)** is dominated by which phase of generation?
* [ ] The decode phase (generating each subsequent token)
* [x] The prefill phase (processing the full prompt to build the KV cache)
* [ ] Tokenizer download
* [ ] Model quantization

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** Prefill processes the entire prompt in parallel and is compute-bound; it sets TTFT. Decode then generates tokens one at a time (memory-bandwidth-bound), setting inter-token latency.
</details>

---

## 🔴 Section 5: Responsible Model Development (Module 06)

**Question 22:** Where does bias in a language model **primarily** originate?
* [ ] From using too few attention heads
* [x] From training data that reflects societal stereotypes and imbalances, amplified by pipeline choices
* [ ] From quantizing the model to 4-bit
* [ ] From using RoPE instead of absolute positions

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** Models learn from human-generated data that encodes stereotypes; data selection, labeling, and optimization can amplify this. Bias is an inherent, data-driven property to be measured and mitigated, not a rare bug.
</details>

---

**Question 23:** What is **red-teaming** in the context of model safety?
* [ ] A load test that measures throughput at high QPS
* [ ] A method to compress model weights
* [x] Deliberately probing a model with adversarial prompts to surface safety failures before deployment
* [ ] A technique to speed up tokenization

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C

**Explanation:** Red-teaming combines expert manual attacks and automated attack generation to find harmful content, bias, jailbreaks, and leakage, then feeds the findings back into mitigations and evaluation.
</details>

---

**Question 24:** Why should a model card report **disaggregated** evaluation metrics rather than a single aggregate score?
* [ ] Aggregate scores are illegal
* [x] A single average can hide large performance or fairness gaps across demographic groups or conditions
* [ ] Disaggregated metrics are always higher
* [ ] It reduces the model's file size

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B

**Explanation:** Disaggregating by group and condition reveals where a model underperforms or is unfair — essential for safety, fairness, honest documentation, and emerging regulatory obligations (e.g., the EU AI Act).
</details>

---

## ⚫ Section 6: End-to-End Lifecycle & Capstone (Module 07)

**Question 25:** In an end-to-end pipeline, why is the model evaluated **both before and after** quantization?
* [ ] To double the reported accuracy
* [ ] Because quantization always improves quality
* [x] The post-quantization evaluation is a regression gate that verifies compression did not degrade quality beyond an acceptable margin
* [ ] To retrain the tokenizer after quantization

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C

**Explanation:** Quantization can silently drop accuracy. Comparing the pre- and post-quantization models on the same suite (within a tolerance) decides whether the compressed model is safe to serve or needs a higher bit-width or a better method.
</details>

---

## 🎓 Capstone Verification & Certification

Complete all hands-on Jupyter Notebook labs and achieve $\ge 80\%$ (20/25) on this final evaluation for SLM Development Certification.

**Score interpretation:**
- **23–25 correct (92–100%):** Distinction — strong end-to-end mastery across all modules.
- **20–22 correct (80–88%):** Pass — certification requirements met.
- **Below 20 (< 80%):** Not yet passing — review the flagged modules and retake.

Use the section breakdown to target review: Architecture (Q1–6), Data & Training (Q7–11), PEFT/LoRA (Q12–16), Optimization & Deployment (Q17–21), Responsible AI (Q22–24), and End-to-End Lifecycle (Q25).
