# Module 5: Optimization and Deployment

Quantize and serve a fine-tuned SLM for production or edge use with minimum latency and memory footprint.

## Lessons

1. [Quantization: Shrinking Models for Edge and CPU Deployment](01_quantization_shrinking_models_for_edge_and_cpu_deployment.md)
   - *~30 minutes*
2. [Serving a Fine-Tuned SLM: API Endpoints and Inference Pipelines](02_serving_a_fine_tuned_slm_api_endpoints_and_inference_pipelines.md)
   - *~35 minutes*
3. [Module Quiz](quiz.md)

## Module Objectives

By the end of this module you will be able to:
- Explain how int8 and int4 quantization reduce model size and memory usage
- Apply bitsandbytes int8 quantization to a fine-tuned SLM and measure size reduction
- Benchmark inference latency and output quality before and after quantization
- Select the appropriate quantization strategy for CPU, mobile, and server targets
- Build a FastAPI inference endpoint that loads a quantized SLM and returns streamed responses
- Configure input length limits and basic output content filtering as production baselines
- Measure time-to-first-token and tokens-per-second for a served SLM
- Push a fine-tuned adapter to the Hugging Face Hub with a complete model card

---

[← Back to Course](../../README.md)