# Module 5 Quiz: Optimization and Deployment

Test your understanding of Optimization and Deployment.

---

## Questions

**Q1: A robotics team needs to run a fine-tuned SLM on a Raspberry Pi 5 with no internet connection. The model currently loads at 1.2 GB in float32. Which quantization strategy gives the best balance of size reduction and inference quality for this constraint?**

- A) GPTQ 8-bit with a GPU calibration dataset
- B) INT4 weight-only quantization using GGUF/llama.cpp
- C) BFloat16 mixed precision with `torch.autocast`
- D) Dynamic range quantization applied only to attention layers

<details>
<summary>Answer</summary>

**Correct: B**

GGUF with llama.cpp targets CPU-only inference and supports INT4 weight-only quantization, reducing a 1.2 GB float32 model to roughly 300–400 MB while maintaining acceptable perplexity. GPTQ (A) requires a CUDA GPU for calibration and runtime. BFloat16 (C) halves memory but still demands hardware with BF16 support, which Raspberry Pi does not provide. Layer-selective dynamic quantization (D) yields marginal size savings and is not an established SLM deployment pattern.

</details>

---

**Q2: You inherit the following FastAPI inference endpoint. A colleague reports that sending a 12,000-token prompt causes the server process to hang indefinitely. Identify the bug and the correct fix.**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
generator = pipeline("text-generation", model="./fine-tuned-slm")

class PromptRequest(BaseModel):
    text: str

@app.post("/generate")
async def generate(request: PromptRequest):
    result = generator(request.text, max_new_tokens=256)
    return {"output": result[0]["generated_text"]}
```

- A) The `pipeline` object is not thread-safe and must be wrapped in `asyncio.Lock`
- B) There is no input length validation; an oversized prompt fills the model's context window and causes OOM-induced hang
- C) `max_new_tokens` must be replaced with `max_length` to prevent infinite generation
- D) The endpoint must use `GET` instead of `POST` for stateless inference

<details>
<summary>Answer</summary>

**Correct: B**

The endpoint accepts arbitrary-length `text` with no guard. A 12,000-token input exceeds most SLMs' context windows (commonly 2,048–4,096 tokens), causing the model to either silently truncate or attempt to allocate memory it cannot satisfy, hanging the process. The fix is to add a `max_input_tokens` check before calling the pipeline and return HTTP 422 if exceeded. Option A is a valid concurrency concern but not the cause of this specific hang. Option C is incorrect — `max_new_tokens` is the preferred parameter in modern Transformers. Option D is architecturally wrong; `POST` is correct for payloads.

</details>

---

**Q3: During post-training static quantization, a calibration dataset is required. What is the calibration dataset's role, and what makes a poor choice?**

- A) It re-trains the model weights; a poor choice is a dataset from a different language
- B) It measures activation ranges so scale factors can be fixed; a poor choice is a dataset that does not represent production inputs
- C) It selects which layers to quantize; a poor choice is a dataset that is too large, causing overfitting
- D) It validates numerical precision against float32 outputs; any held-out split of the training set is ideal

<details>
<summary>Answer</summary>

**Correct: B**

Static quantization records the minimum and maximum activation values per tensor across the calibration set to compute fixed scale factors used at inference time. If the calibration data is unrepresentative — for example, short generic sentences when the production workload is long legal clauses — the fixed scales will be wrong, leading to clipping and accuracy degradation. Option A conflates calibration with fine-tuning. Option C describes a different process (structured pruning). Option D is partially true in spirit but wrong in detail: using only the training split risks distribution mismatch versus real traffic.

</details>

---

**Q4: A startup serves a 360M-parameter SLM with a single-worker Uvicorn process. Under load testing at 20 concurrent users, median latency jumps from 180 ms to 4.2 seconds. Which two changes will most directly address this bottleneck? (Select the best single answer that names both.)**

- A) Switch to INT4 quantization and increase `max_new_tokens` to reduce generation loops
- B) Add Gunicorn with multiple Uvicorn workers and enable response streaming via Server-Sent Events
- C) Replace FastAPI with Flask and pin the model to CPU to reduce GPU memory contention
- D) Enable `torch.compile` on the model and increase the Pydantic validation timeout

<details>
<summary>Answer</summary>

**Correct: B**

A single Uvicorn worker processes one request at a time; 20 concurrent users queue behind each other, multiplying latency. Adding Gunicorn with multiple workers (or async batching) parallelizes request handling. Streaming via SSE returns the first tokens immediately, cutting *perceived* latency even when total generation time is unchanged. Option A reduces model size but does not fix the single-worker bottleneck. Option C regresses performance — Flask is synchronous and CPU pinning removes GPU throughput. Option D's `torch.compile` helps per-request speed marginally but cannot compensate for a serialized request queue.

</details>

---

**Q5: You quantize a fine-tuned SLM to INT4 and evaluate it on your domain-specific test set. ROUGE-L drops from 0.61 (float32 baseline) to 0.43. Before reverting to float32, what is the most targeted next step?**

- A) Re-run fine-tuning from scratch on the quantized model weights
- B) Switch from weight-only INT4 to INT8 activation-plus-weight quantization, or apply GPTQ with a domain-representative calibration set
- C) Increase `temperature` during inference to recover output diversity lost during quantization
- D) Add a post-processing regex filter to correct common quantization artifacts in generated text

<details>
<summary>Answer</summary>

**Correct: B**

A large ROUGE-L drop under INT4 usually signals that aggressive weight compression is losing precision critical to the domain's vocabulary distribution. Stepping up to INT8 or using GPTQ — which minimizes quantization error layer-by-layer using second-order gradient information — recovers much of this accuracy at a modest size penalty. Option A (fine-tuning quantized weights) is non-standard and unsupported by most frameworks without quantization-aware training (QAT) setup. Option C changes generation behavior, not numerical precision. Option D treats symptoms rather than root cause and will not scale.

</details>

---

**Q6: Reflection prompt — answer in 3–5 sentences.**

You are advising a clinical documentation team that wants to deploy a fine-tuned SLM on hospital workstations (Windows 10, no GPU, 8 GB RAM). They ask whether quantization alone makes their deployment safe and production-ready. What would you tell them, and what additional safeguards would you recommend?

- A) Yes — quantization reduces model size and therefore reduces all operational risk
- B) No — quantization addresses memory and latency constraints but does not constitute a safety or reliability solution on its own
- C) It depends entirely on the quantization format chosen; GGUF models are inherently safer than GPTQ models
- D) Quantization is unnecessary if the model is already fine-tuned on clinical data

<details>
<summary>Answer</summary>

**Correct: B**

Quantization solves the compute problem — fitting the model into 8 GB RAM on a CPU-only machine — but says nothing about output safety, hallucination rates, or compliance. For a clinical setting, the team also needs: output length limits and input sanitization at the API layer, a structured output schema (e.g., JSON with constrained fields) to reduce free-form hallucination, human-in-the-loop review for any generated clinical text, and an audit log of inputs and outputs for regulatory traceability. Quantization format (C) is irrelevant to safety properties. Fine-tuning on domain data (D) improves relevance but does not eliminate hallucination risk.

</details>

---

## Capstone Challenge

You have a 135M-parameter SLM fine-tuned on customer support tickets for a European logistics company. The task is to deploy this model as a production inference API that runs on a CPU-only cloud VM (4 vCPUs, 16 GB RAM) and returns structured JSON responses (issue category + suggested resolution) within 1.5 seconds at the 95th percentile for inputs up to 512 tokens.

Complete all of the following:

1. **Quantize** the model to a format appropriate for CPU deployment. Document the calibration dataset you used, why you chose it, and report the ROUGE-L or F1 score before and after quantization.
2. **Build a FastAPI service** that accepts a ticket string, validates input length, runs inference, and returns a validated Pydantic response model containing `issue_category: str` and `suggested_resolution: str`.
3. **Load test** the endpoint using `locust` or `wrk` at 10 concurrent users. Record median and p95 latency. If p95 exceeds 1.5 seconds, apply one optimization (batching, streaming, or worker scaling) and re-test.
4. **Security and safety audit**: Send five adversarial inputs (e.g., prompt injection attempts, inputs in a non-supported language, inputs exceeding 512 tokens) and document the observed behavior and the guardrails you added.

**Evaluation Rubric:**

- **Quantization quality**: Excellent means INT4 or INT8 GGUF/GPTQ format selected with justification, calibration set drawn from held-out support tickets, and ROUGE-L or F1 degradation under 5 percentage points versus float32 baseline.
- **API correctness**: Excellent means the endpoint enforces a 512-token input limit with HTTP 422 on violation, returns a valid Pydantic-typed JSON body on every successful request, and handles pipeline exceptions with HTTP 500 and a structured error message — never a raw Python traceback.
- **Latency target**: Excellent means p95 latency at 10 concurrent users is at or below 1.5 seconds, documented with a load test report (tool used, concurrency level, result table), and any optimization applied is explained with before/after numbers.
- **Safety audit**: Excellent means all five adversarial inputs are tested, each response is documented, at least two guardrails are implemented (e.g., input length cap, output schema enforcement), and a one-paragraph statement acknowledges what fine-tuning and quantization do *not* protect against.