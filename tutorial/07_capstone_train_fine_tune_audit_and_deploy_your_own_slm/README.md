# Module 7: Capstone: Train, Fine-Tune, Audit, and Deploy Your Own SLM

Integrate every skill from the course into a single end-to-end project: build a domain-specific <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLM</abbr> from a <abbr title="A model trained on a massive general dataset to learn language patterns before fine-tuning.">pretrained</abbr> checkpoint, fine-tune it with <abbr title="Low-Rank Adaptation: an efficient fine-tuning method that freezes base model weights and injects small trainable adapter matrices.">LoRA</abbr>, audit it for bias, quantize it, and serve it via an API endpoint.

## Lessons

1. [Capstone Project Briefing and Dataset Selection](01_capstone_project_briefing_and_dataset_selection.md)
   - *~20 minutes*
2. [End-to-End Build: Fine-Tune, Quantize, Audit, and Serve](02_end_to_end_build_fine_tune_quantize_audit_and_serve.md)
   - *~90 minutes*

## Module Objectives

By the end of this module you will be able to:
- Select a target domain and source a suitable public dataset from the Hugging Face Hub
- Define a concrete task (text classification, Q&A, summarization, or code completion) with success criteria
- Outline a project plan covering data preparation, <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">fine-tuning</abbr>, evaluation, bias audit, and deployment steps
- Identify the compute tier available and select an appropriate SLM checkpoint accordingly
- Complete the full pipeline: data prep → LoRA fine-tuning → <abbr title="Recall-Oriented Understudy for Gisting Evaluation: metrics evaluating summary quality by comparing against human references.">ROUGE</abbr>/F1 evaluation → bias audit → int8 <abbr title="The process of reducing weight precision (e.g. from 16-bit to 4-bit) to shrink model size and speed up inference.">quantization</abbr> → FastAPI serving → model card publication
- Achieve evaluation targets defined in the capstone rubric (<abbr title="A metric measuring how well a probability model predicts a sample; lower perplexity indicates higher confidence and quality.">perplexity</abbr>, ROUGE-L, latency)
- Document at least three adversarial prompt failure modes in the bias audit log
- Publish the final adapter, quantized model, and model card to the Hugging Face Hub

---

[← Back to Course](../../README.md)