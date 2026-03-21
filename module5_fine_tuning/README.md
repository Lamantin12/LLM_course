# Module 5 — Fine-tuning & Open-Source Models

Survey of open-source LLMs and hands-on fine-tuning of Llama 3.1 with LoRA/QLoRA using the Unsloth framework.

> **Note:** GPU required (Google Colab recommended). The zoo notebook cannot be run start-to-finish in a single session — different models require different resources, run each section independently. The fine-tuning notebooks require a Colab A100/T4 runtime.

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
- When to use fine-tuning vs RAG — consistency, domain adaptation, privacy, context limits
- Instruction fine-tuning — Alpaca prompt format (Instruction / Input / Response), EOS token
- Dataset preparation — exporting Telegram posts (JSON), cleaning, LLM-assisted topic labeling, uploading to HuggingFace
- PEFT (Parameter-Efficient Fine-Tuning) — updating 1–10% of parameters instead of full model
- LoRA — low-rank adapter matrices trained on top of frozen base weights; key params: `r`, `target_modules`, `lora_alpha`
- QLoRA — combining 4-bit quantization with LoRA to fine-tune on consumer GPUs
- Unsloth — fine-tuning framework giving 2–5x speedup, lower VRAM; patches HuggingFace transformers
- SFTTrainer (TRL) — supervised fine-tuning trainer; key args: batch size, gradient accumulation, learning rate, `max_steps`
- Saving LoRA adapters — `save_pretrained` (local) and `push_to_hub` (HuggingFace); only adapter weights saved (~168 MB)
- RAFT — RAG + Fine-Tuning hybrid for document-grounded generation

---

## Files

| File | Description |
|------|-------------|
| `M5_Zoo.ipynb` | Lecture notebook — model zoo survey with runnable examples for each topic |
| `M5_2_Dataset_prepare.ipynb` | Lecture notebook — preparing a fine-tuning dataset from Telegram posts: JSON parsing, LLM-assisted labeling, HuggingFace upload |
| `M5_2_FineTuning.ipynb` | Lecture notebook — fine-tuning Llama 3.1 8B with Unsloth, QLoRA/LoRA, SFTTrainer; inference before/after; saving LoRA adapters |

---

## How to Run

```bash
# Model zoo
pip install transformers bitsandbytes accelerate langchain-community llama-cpp-python pillow torch
jupyter notebook module5_fine_tuning/M5_Zoo.ipynb

# Fine-tuning (run on Google Colab with GPU)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes triton datasets
jupyter notebook module5_fine_tuning/M5_2_Dataset_prepare.ipynb
jupyter notebook module5_fine_tuning/M5_2_FineTuning.ipynb
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
- **Instruction fine-tuning**: SFT on (instruction, input, response) triplets to teach a base model to follow directives.
- **Alpaca prompt**: Standard prompt template for instruction fine-tuning with Instruction / Input / Response sections; requires an EOS token appended to each sample.
- **LoRA (Low-Rank Adaptation)**: Adds small trainable rank-`r` matrices alongside frozen base weights; only adapter parameters are updated and saved.
- **QLoRA**: Combines 4-bit quantized base model loading (`load_in_4bit`) with LoRA adapters — enables fine-tuning large models on consumer GPUs.
- **PEFT**: Family of methods (LoRA, prefix tuning, etc.) that update a small fraction of model parameters to avoid full fine-tuning cost.
- **Unsloth**: HuggingFace-compatible fine-tuning framework that patches attention kernels for 2–5x faster training and lower VRAM usage.
- **SFTTrainer**: TRL trainer for supervised fine-tuning; accepts a `dataset_text_field` and wraps HuggingFace `TrainingArguments`.
- **Catastrophic forgetting**: Risk that fine-tuning on narrow domain data degrades general-purpose capabilities; mitigated by PEFT/LoRA.
- **RAFT**: RAG + Fine-Tuning — fine-tunes a model to select and use retrieved documents, combining retrieval accuracy with generation quality.
