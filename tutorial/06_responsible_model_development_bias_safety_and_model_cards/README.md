# Module 6: Responsible Model Development: Bias, Safety, and Model Cards

Audit a fine-tuned <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLM</abbr> for bias and failure modes, apply baseline safety mitigations, and document the model with a complete model card.

## Lessons

1. [Bias Detection and Safety Auditing for SLMs](01_bias_detection_and_safety_auditing_for_slms.md)
   - *~30 minutes*
2. [Writing a Model Card and Communicating Model Limitations](02_writing_a_model_card_and_communicating_model_limitations.md)
   - *~25 minutes*
3. [Module Quiz](quiz.md)

## Module Objectives

By the end of this module you will be able to:
- Run a structured set of adversarial and demographic prompts to surface failure modes
- Document observed outputs in a bias audit log with severity ratings
- Explain why <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">fine-tuning</abbr> alone cannot eliminate harmful outputs
- Apply at least one mitigation: output filtering, refusal prompting, or constrained decoding
- Author a complete Hugging Face model card covering intended use, training data, evaluation results, and known limitations
- Include bias audit findings and recommended mitigations in the model card
- Apply the model card framework citing Mitchell et al. (2019)
- Publish the model card alongside the adapter weights on the Hugging Face Hub

---

[← Back to Course](../../README.md)