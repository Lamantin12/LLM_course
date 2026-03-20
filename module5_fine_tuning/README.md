# Module 5 — Open-Source Model Zoo

A survey of open-source LLMs: how to choose, load, and run them locally or on Colab — with quantization, MoE, Russian-language models, and multimodal models.

> **Note:** This notebook requires a GPU (Google Colab recommended). It cannot be run start-to-finish in a single session — different models require different resources. Run each section independently.

---

## Topics Covered

- Foundation model variants — base, chat/instruct, and code models; what changes during fine-tuning
- Benchmarks — MMLU, HumanEval, and how to read leaderboard scores
- Choosing a model by task type — base vs. instruct vs. code
- Quantization with `bitsandbytes` — 4-bit and 8-bit loading for consumer GPUs
- Mixture of Experts (MoE) — sparse models that activate only a subset of parameters per token
- Russian-language models — running quantized GGUF models locally via `LlamaCpp`
- Multimodal models — vision-language models that accept both images and text
- Licenses — Apache 2.0, Llama Community License, MIT; commercial-use restrictions

---

## Files

| File | Description |
|------|-------------|
| `M5_Zoo.ipynb` | Lecture notebook — model zoo survey with runnable examples for each topic |

---

## How to Run

```bash
pip install transformers bitsandbytes accelerate langchain-community llama-cpp-python pillow torch
jupyter notebook module5_fine_tuning/M5_Zoo.ipynb
```

Models are downloaded from HuggingFace Hub on first run. GGUF models must be downloaded manually and the path passed to `LlamaCpp`.

---

## Key Concepts

- **Base model**: Pretrained on raw text — no instruction following; use for fine-tuning or completion tasks.
- **Instruct/Chat model**: Further trained with RLHF or SFT to follow instructions and hold conversations.
- **Code model**: Trained on code corpora (e.g., CodeLlama, StarCoder) — optimised for generation and completion of code.
- **Quantization**: Reducing weight precision (32-bit → 8-bit → 4-bit) to fit large models on consumer GPUs; trades slight quality for memory savings.
- **`BitsAndBytesConfig`**: HuggingFace config object for loading models in 4-bit or 8-bit precision via `bitsandbytes`.
- **MoE (Mixture of Experts)**: Sparse architecture where only a fraction of parameters are activated per token — faster inference at equivalent quality to dense models of similar total size.
- **GGUF**: Quantized model format for CPU/GPU inference via `llama.cpp`; must be loaded with `LlamaCpp`, not the HuggingFace pipeline.
- **`pipeline` (HuggingFace)**: High-level API that handles tokenisation, model call, and decoding in one call — supports `"text-generation"`, `"image-text-to-text"`, and more.
