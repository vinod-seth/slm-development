# Module 3: Data Preparation and Training from Scratch

Build a complete data pipeline using Hugging Face Datasets and run a minimal training loop on a small domain-specific corpus.

## Lessons

1. [Building a Domain-Specific Dataset with Hugging Face Datasets](01_building_a_domain_specific_dataset_with_hugging_face_datasets.md)
   - *~30 minutes*
2. [Writing a Training Loop: Trainer API and Manual PyTorch](02_writing_a_training_loop_trainer_api_and_manual_pytorch.md)
   - *~40 minutes*
3. [Evaluating Your Trained Model: Perplexity, ROUGE, and F1](03_evaluating_your_trained_model_perplexity_rouge_and_f1.md)
   - *~25 minutes*
4. [Module Quiz](quiz.md)

## Module Objectives

By the end of this module you will be able to:
- Load a public dataset from the Hugging Face Hub and inspect its schema
- Write a preprocessing function to tokenize text with dynamic padding and truncation
- Apply dataset.map() with batched processing to tokenize at scale
- Split a dataset into train, validation, and test partitions with reproducible seeds
- Configure a TrainingArguments object with learning rate, batch size, and evaluation strategy
- Train a SmolLM2-135M model on a custom dataset using the Hugging Face Trainer
- Write a minimal manual PyTorch training loop to understand what Trainer abstracts away
- Compute <abbr title="A metric measuring how well a probability model predicts a sample; lower perplexity indicates higher confidence and quality.">perplexity</abbr> on a held-out validation set and interpret the result
- Evaluate a trained model using perplexity, <abbr title="Recall-Oriented Understudy for Gisting Evaluation: metrics evaluating summary quality by comparing against human references.">ROUGE</abbr>-L, and F1 metrics on a held-out test set
- Use the evaluate library to compute ROUGE scores on a generation task
- Interpret a worked ROUGE and F1 example end-to-end
- Identify common failure modes: <abbr title="A training error where a model learns training data details too well, performing poorly on new data.">overfitting</abbr>, catastrophic forgetting, and distribution mismatch

---

[← Back to Course](../../README.md)