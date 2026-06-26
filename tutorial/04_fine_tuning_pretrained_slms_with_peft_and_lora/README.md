# Module 4: Fine-Tuning Pretrained SLMs with PEFT and LoRA

Apply parameter-efficient <abbr title="Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset.">fine-tuning</abbr> techniques to adapt a <abbr title="A model trained on a massive general dataset to learn language patterns before fine-tuning.">pretrained</abbr> <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLM</abbr> to a target task using a fraction of the compute required for full fine-tuning.

## Lessons

1. [Why Pretrained Weights Accelerate Fine-Tuning](01_why_pretrained_weights_accelerate_fine_tuning.md)
   - *~20 minutes*
2. [LoRA Fine-Tuning: Theory and Implementation](02_lora_fine_tuning_theory_and_implementation.md)
   - *~40 minutes*
3. [Instruction Fine-Tuning for Task-Specific SLMs](03_instruction_fine_tuning_for_task_specific_slms.md)
   - *~35 minutes*
4. [Module Quiz](quiz.md)

## Module Objectives

By the end of this module you will be able to:
- Explain how pretrained weights encode transferable language knowledge and reduce data requirements
- Compare memory footprint of full fine-tuning vs <abbr title="Parameter-Efficient Fine-Tuning: techniques (like LoRA) that adapt pre-trained models by updating only a tiny fraction of parameters.">PEFT</abbr> fine-tuning for a 135M model
- Identify which layers benefit most from task-specific adaptation
- Confirm successful completion of Module 3 training runs before proceeding
- Describe the <abbr title="Low-Rank Adaptation: an efficient fine-tuning method that freezes base model weights and injects small trainable adapter matrices.">LoRA</abbr> low-rank decomposition technique and explain why it reduces trainable parameters — citing Hu et al. (2021)
- Configure a LoraConfig targeting attention projection layers with rank, alpha, and dropout
- Apply get_peft_model() to wrap a pretrained SLM and inspect trainable parameter count
- Train a LoRA adapter on a classification or generation task and save the adapter weights
- Format a dataset into instruction-response pairs using a prompt template
- Fine-tune a LoRA-wrapped SLM on an instruction dataset with the Trainer API
- Merge adapter weights back into the base model and verify output quality
- Document known safety limitations of instruction fine-tuning without RLHF

---

[← Back to Course](../../README.md)