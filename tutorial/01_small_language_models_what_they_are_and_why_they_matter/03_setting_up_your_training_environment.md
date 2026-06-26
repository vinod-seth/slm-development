# Setting Up Your Training Environment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-seth/slm-development/blob/main/tutorial/01_small_language_models_what_they_are_and_why_they_matter/03_setting_up_your_environment.ipynb)

| | |
|---|---|
| **Domain** | GenAI |
| **Module** | Small Language Models: What They Are and Why They Matter |
| **Difficulty** | Beginner |
| **Estimated Time** | 25 minutes |
| **Prerequisites** | Basic Python programming knowledge; familiarity with what a model is and the difference between training and <abbr title="Running a trained model to generate predictions or text output from new, unseen inputs.">inference</abbr>; no prior deep learning or NLP experience required |

---

## Lesson Roadmap

- **🟢 Core Concepts** — Why dependency isolation prevents silent training failures, and how the course repo is organized.
- **🔷 Technical Deep-Dive** — Create a virtual environment, install the pinned stack, verify your <abbr title="Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning.">GPU</abbr> (or <abbr title="Central Processing Unit: the general-purpose processor in a computer.">CPU</abbr> fallback), and run your first five-line diagnostic inside 10 minutes.
- **🔷 DevContainer / Dockerfile** — Zero-configuration cloud setup for learners without a local CUDA-capable GPU.
- **🔷 Hugging Face Hub Setup** — Create an account and authenticate the CLI.
- **🧪 Hands-On Exercise** — Confirm every dependency version matches the compatibility matrix and commit the result to your repo.

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Create an isolated Python virtual environment and confirm the correct PyTorch and Transformers versions are installed.
- Verify GPU availability — or configure a CPU fallback — using a five-line diagnostic script.
- Navigate the course repository directory structure and locate each module folder.
- Use the provided devcontainer or Dockerfile for zero-friction cloud setup.

---

## 🟢 Core Concepts

### How Conflicting Dependencies Break Training Runs

A Python environment is a container for packages and their exact versions. Without isolation, installing two projects side by side almost guarantees a conflict. For <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLM</abbr> training, the failure mode is subtle: PyTorch silently falls back to CPU when a mismatched CUDA version breaks the GPU bridge. You won't see an error — you'll just see your training run take 40× longer than expected.

Think of a virtual environment like a separate toolbox for each project. The tools in one box never interfere with another box, even if two boxes contain different versions of the same wrench.

The course uses **Python 3.11** and **CUDA 12.1** throughout. The table below is your compatibility anchor.

| Component | Pinned Version | Last verified |
|---|---|---|
| Python | 3.11.x | 2025-01 |
| PyTorch | 2.2.x (cu121) | 2025-01 |
| Hugging Face Transformers | 4.40.x | 2025-01 |
| <abbr title="Parameter-Efficient Fine-Tuning: techniques (like LoRA) that adapt pre-trained models by updating only a tiny fraction of parameters.">PEFT</abbr> | 0.10.x | 2025-01 |
| Datasets (HF) | 2.19.x | 2025-01 |
| Accelerate | 0.29.x | 2025-01 |

> [!IMPORTANT]
> These versions are reviewed quarterly. Check `CHANGELOG.md` in the course repository for any updates since this lesson was published.

### Course Repository Layout

```
slm-training-course/
├── README.md
├── CHANGELOG.md
├── metadata.json
├── .devcontainer/
│   └── devcontainer.json       # VS Code devcontainer spec
├── docker/
│   └── Dockerfile              # CUDA 12.1 + Python 3.11 image
├── assets/
│   └── diagrams/
├── modules/
│   ├── 01_introduction/
│   │   ├── README.md
│   │   ├── lesson1.md
│   │   ├── lesson2.md
│   │   └── lesson3.md          ← you are here
│   ├── 02_tokenization/
│   ├── 03_training_fundamentals/
│   ├── 04_fine_tuning/
│   ├── 05_optimization_deployment/
│   ├── 06_responsible_development/
│   └── 07_capstone/
└── envs/
    └── requirements.txt
```

Each module folder contains a `README.md` listing its prerequisites and estimated cumulative time from course start. Navigate there before starting any lesson.

> [!NOTE]
> If you are a complete beginner to virtual environments, complete this lesson (Lesson 3) before running any install commands shown in Lesson 1 or Lesson 2.

---

## 🔷 Technical Deep-Dive

### Step 1 — Create and Activate a Virtual Environment

**macOS / Linux (bash or zsh):**

```bash
python3.11 -m venv slm_env
source slm_env/bin/activate
```

**Windows (CMD — not PowerShell):**

```cmd
py -3.11 -m venv slm_env
slm_env\Scripts\activate.bat
```

> [!IMPORTANT]
> On Windows, use **CMD** (`cmd.exe`), not PowerShell. PowerShell requires an execution-policy change (`Set-ExecutionPolicy RemoteSigned`) that is outside the scope of this course and varies by corporate policy.

Confirm activation — your prompt should now show `(slm_env)`.

---

### Step 2 — Install the Pinned Stack

```bash
# Upgrade pip first to avoid resolver warnings
python -m pip install --upgrade pip

# PyTorch with CUDA 12.1 support
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Core training stack — pinned for reproducibility
pip install \
    transformers==4.40.2 \
    datasets==2.19.1 \
    peft==0.10.0 \
    accelerate==0.29.3 \
    sentencepiece==0.2.0 \
    huggingface_hub==0.23.0

# Notebook support
pip install jupyterlab==4.1.6 ipywidgets==8.1.2
```

> [!NOTE]
> CPU-only machines: replace the `torch` install line with `pip install torch==2.2.2`. All later lessons detect the device automatically — no other change is needed.

---

### Step 3 — Five-Line GPU Diagnostic *(run this within the first 10 minutes)*

Create `diagnostics/check_env.py` in your repo and run it immediately after installation.

```python
# diagnostics/check_env.py
# Purpose: confirm GPU availability and correct package versions.
# Run: python diagnostics/check_env.py

import sys
import torch
import transformers
import datasets
import peft

EXPECTED = {
    "python": (3, 11),
    "torch": "2.2.2",
    "transformers": "4.40.2",
    "datasets": "2.19.1",
    "peft": "0.10.0",
}

def check_python_version(expected_major: int, expected_minor: int) -> None:
    major, minor = sys.version_info.major, sys.version_info.minor
    status = "✅" if (major, minor) == (expected_major, expected_minor) else "❌"
    print(f"{status}  Python {major}.{minor} (expected {expected_major}.{expected_minor})")

def check_package_version(name: str, installed: str, expected: str) -> None:
    status = "✅" if installed.startswith(expected[:5]) else "❌"
    print(f"{status}  {name} {installed} (expected {expected})")

def check_gpu() -> None:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✅  GPU detected: {device_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("⚠️   No GPU detected — CPU fallback active. Training will be slower.")

if __name__ == "__main__":
    print("=== SLM Training Environment Diagnostic ===\n")
    check_python_version(*EXPECTED["python"])
    check_package_version("torch",          torch.__version__,          EXPECTED["torch"])
    check_package_version("transformers",   transformers.__version__,   EXPECTED["transformers"])
    check_package_version("datasets",       datasets.__version__,       EXPECTED["datasets"])
    check_package_version("peft",           peft.__version__,           EXPECTED["peft"])
    print()
    check_gpu()
    print("\n=== Diagnostic complete ===")
```

**Expected output (GPU machine):**

```
=== SLM Training Environment Diagnostic ===

✅  Python 3.11 (expected 3.11)
✅  torch 2.2.2+cu121 (expected 2.2.2)
✅  transformers 4.40.2 (expected 4.40.2)
✅  datasets 2.19.1 (expected 2.19.1)
✅  peft 0.10.0 (expected 0.10.0)

✅  GPU detected: NVIDIA RTX 3060 (12.0 GB VRAM)

=== Diagnostic complete ===
```

Any `❌` line means a version mismatch. Re-run the pip install block from Step 2 with `--force-reinstall` for the flagged package.

---

### Step 4 — DevContainer / Dockerfile (Cloud or Zero-Config Setup)

If you don't have a local CUDA-capable GPU, use the provided Docker image. It targets **Python 3.11 + CUDA 12.1** and pre-installs the full pinned stack.

**`docker/Dockerfile`:**

```dockerfile
# Base image: NVIDIA CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3-pip \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /workspace

# Copy and install pinned requirements
COPY envs/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r /workspace/requirements.txt

# Default to bash for interactive container sessions
CMD ["/bin/bash"]
```

**Build and run:**

```bash
docker build -t slm-training:1.0 -f docker/Dockerfile .
docker run --gpus all -it --rm \
    -v "$(pwd)":/workspace \
    slm-training:1.0
```

**VS Code DevContainer** users: open the repo folder, select **"Reopen in Container"**, and VS Code reads `.devcontainer/devcontainer.json` automatically. The container build runs once; subsequent opens take under 10 seconds.

---

### Step 5 — Hugging Face Hub Account and CLI Authentication

Several models used in later modules (starting in Module 4) require a Hugging Face account to accept gating terms.

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Navigate to **Settings → Access <abbr title="A sub-word unit, word, or character that text is split into for processing by a language model.">Tokens</abbr>** and create a token with **read** scope.
3. Authenticate the CLI:

```bash
# Store the token securely — never hardcode it in scripts
huggingface-cli login
# Paste your token at the prompt; it is saved to ~/.cache/huggingface/token
```

Confirm authentication:

```bash
huggingface-cli whoami
# Expected output: your HF username
```

> [!IMPORTANT]
> Never commit your HF token to version control. Add `~/.cache/huggingface/` to `.gitignore` and use environment variables (`HF_TOKEN`) in CI pipelines.

---

## Hands-On Exercise

**Goal:** Produce a verified environment snapshot and save it to your repo.

**Steps:**

1. Activate `slm_env` (or start your devcontainer).
2. Run `python diagnostics/check_env.py`. All five version lines must show `✅`.
3. Export your exact environment:

```bash
pip freeze > envs/verified_environment_$(date +%Y%m).txt
```

4. Open `envs/verified_environment_2025XX.txt` and confirm `torch==2.2.2+cu121` (or `torch==2.2.2` for CPU) appears in the list.
5. Add, commit, and push the file:

```bash
git add envs/verified_environment_*.txt diagnostics/check_env.py
git commit -m "feat: add environment diagnostic and verified snapshot"
git push origin main
```

**Verifiable outcome:** Your repository now contains a timestamped dependency snapshot. Any teammate can reproduce your exact environment by running `pip install -r envs/verified_environment_2025XX.txt`.

> [!NOTE]
> This snapshot is also your rollback point if a future `pip install` breaks something. Keep one snapshot per quarter, matching the course's quarterly review cycle.

---

## Concept Check

**Question 1 — Version Compatibility**

Which Python and PyTorch versions does this course pin, and why does the combination matter?

* [ ] Python 3.10 + PyTorch 2.0 — the most widely deployed combination at course launch.
* [x] Python 3.11 + PyTorch 2.2 (cu121) — tested together against CUDA 12.1 to guarantee GPU availability.
* [ ] Python 3.12 + PyTorch 2.3 — the latest available at time of writing.
* [ ] Any Python 3.x + PyTorch 1.x — version pinning is optional for beginner courses.

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option B.

**Explanation:**
PyTorch GPU support depends on a three-way match: Python version, PyTorch version, and the CUDA toolkit version. Python 3.11 + PyTorch 2.2 + CUDA 12.1 is the combination verified in the course compatibility matrix. Using any other combination risks silent CPU fallback — PyTorch loads successfully but ignores the GPU, making training appear to work while running 40× slower.

</details>

---

**Question 2 — Windows Activation**

A colleague on Windows runs `slm_env\Scripts\Activate.ps1` in PowerShell and gets a security error. What is the correct fix?

* [ ] Reinstall Python 3.11 from the Microsoft Store.
* [ ] Run `pip install pywin32` to add PowerShell support.
* [x] Use `slm_env\Scripts\activate.bat` in **CMD** (`cmd.exe`) instead of PowerShell.
* [ ] Add `--user` to the pip install command.

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C.

**Explanation:**
PowerShell enforces an execution policy that blocks unsigned scripts by default. The `.bat` activation script runs without policy restrictions inside `cmd.exe`. Changing the execution policy (`Set-ExecutionPolicy`) requires administrator rights and varies by corporate environment, so the course avoids it entirely.

</details>

---

**Question 3 — Debugging a Diagnostic Failure (scenario-based)**

You run `check_env.py` and see:

```
❌  peft 0.9.0 (expected 0.10.0)
✅  GPU detected: NVIDIA RTX 4070 (12.0 GB VRAM)
```

What is the single most targeted command to fix this without disturbing other packages?

* [ ] `pip install peft` — installs the latest available version.
* [ ] Delete and recreate `slm_env` from scratch.
* [x] `pip install --force-reinstall peft==0.10.0`
* [ ] `pip install peft==0.10.0 --upgrade torch` — upgrades both simultaneously.

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct Answer:** Option C.

**Explanation:**
`--force-reinstall` replaces the installed version with the exact pinned version without touching other packages. Option A installs the *latest* version, which may not be 0.10.0. Option B is destructive and wastes time. Option D combines two operations unnecessarily and risks creating a new mismatch with the torch CUDA wheel.

</details>

---

**Reflection Prompt (open-ended)**

Consider a project where your model needs to run on a hospital's air-gapped (offline) server with no internet access. Which parts of this setup lesson would need to change? Think about package installation, Hugging Face authentication, and the Docker build process.

<details>
<summary>🔑 Suggested Thinking Framework</summary>

Key areas to address:

- **Package installation**: You'd need to pre-download all wheels and use `pip install --no-index --find-links /local/wheels` pointing to a local mirror.
- **HF authentication**: Air-gapped environments can't reach `huggingface.co`. You'd pre-download model weights and use `model = AutoModel.from_pretrained("/local/model_path")` with no network call.
- **Docker build**: The `Dockerfile` would need a multi-stage build where the first stage downloads everything on an internet-connected machine, and the final image is exported and transferred physically.
- **CUDA drivers**: The host machine's CUDA driver version must match the image's CUDA toolkit version — this requires physical verification, not a download.

There is no single correct answer. The goal is to identify which steps assume internet access and design workarounds for each.

</details>

---

## Summary

- **Isolate dependencies first.** A virtual environment (`venv` or the provided Docker image) prevents version conflicts that cause silent GPU fallback — the hardest failure mode to debug in SLM training.
- **Pin every version and record it.** The compatibility matrix (Python 3.11, PyTorch 2.2+cu121, Transformers 4.40.x) is your source of truth. Export a dated snapshot with `pip freeze` after every verified install. Review versions quarterly.
- **Run the diagnostic before writing any training code.** Five lines confirm Python version, all package versions, and GPU availability. A `✅` on every line means your environment is reproducible by anyone on your team.
- **HF Hub authentication is gated from Module 4 onward.** Create your account and store your token as an environment variable now to avoid interruptions later.
- **The devcontainer removes hardware friction.** Learners without a local GPU can build `docker/Dockerfile` once and get an identical, pre-verified environment in the cloud.

---

## References & Credits

- PyTorch Installation Selector — official compatibility matrix for CUDA/Python combinations: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) *(Last verified: 2025-01)*
- Hugging Face Hub documentation — token authentication and CLI reference: [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens) *(Last verified: 2025-01)*
- NVIDIA CUDA 12.1 release notes — driver and toolkit compatibility: [https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) *(Last verified: 2025-01)*
- Hu et al. (2021) *<abbr title="Low-Rank Adaptation: an efficient fine-tuning method that freezes base model weights and injects small trainable adapter matrices.">LoRA</abbr>: Low-Rank Adaptation of Large Language Models.* [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) — The PEFT library installed in this lesson implements this method; the full technique is covered in Module 4.
- PEFT library (Apache 2.0 License) — Hugging Face: [https://github.com/huggingface/peft](https://github.com/huggingface/peft) *(Last verified: 2025-01)*
- Hugging Face Transformers library (Apache 2.0 License): [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers) *(Last verified: 2025-01)*