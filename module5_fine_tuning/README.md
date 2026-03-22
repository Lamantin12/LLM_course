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

## Lecture Notes

### M5_Zoo.ipynb

**Model variants** — The three deployment categories: base (pretrained), instruct/chat (RLHF/SFT), and code (code corpus).
Base models produce continuations, not answers — unsuitable for dialogue without further fine-tuning. Instruct models follow directives; code models specialise in generation and completion of code. Choosing the wrong variant is a common source of poor results — always check the model card for the training regime.
```python
# Base model — continues text, does not answer questions
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-2-7b-hf")       # base
# Instruct model — follows instructions
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-2-7b-chat-hf")  # instruct
```

**Quantization** — Reduces weight precision (32-bit → 8-bit → 4-bit) to fit large models in limited GPU memory.
`BitsAndBytesConfig(load_in_4bit=True)` passed to `AutoModelForCausalLM.from_pretrained` halves or quarters VRAM usage; a 7B model requiring 14 GB at float16 fits in ~4 GB at 4-bit. The quality trade-off is slight — 4-bit quantized models score within 1–2% of full-precision on standard benchmarks.
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config, device_map="auto")
```

**MoE (Mixture of Experts)** — Sparse architecture that routes each token to a subset of expert feed-forward layers.
A Mixtral-8x7B model has 47B total parameters but activates only ~13B per token — the router selects 2 of 8 experts per layer, giving quality comparable to a 47B dense model at the inference speed of a 13B model. The non-obvious cost: all expert weights must fit in memory even though only a fraction are used per forward pass.
```python
# Load Mixtral MoE — requires ~90 GB VRAM full precision; use quantized in practice
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1", quantization_config=bnb_config)
```

**LlamaCpp** — Loads GGUF-format quantized models for CPU or hybrid CPU/GPU inference via `llama.cpp`.
`LlamaCpp(model_path="/path/to/model.gguf", n_ctx=2048, n_gpu_layers=40)` reads the pre-quantized binary; `n_gpu_layers` controls how many transformer layers are offloaded to GPU. GGUF files must be downloaded manually — the HuggingFace `pipeline` API cannot load them.
```python
from langchain_community.llms import LlamaCpp
llm = LlamaCpp(
    model_path="/models/saiga_llama3_8b_q4.gguf",
    n_ctx=2048, n_gpu_layers=40, verbose=False)
print(llm.invoke("Привет! Кто ты?"))
```

**Multimodal** — Vision-language models that accept both image and text inputs and generate text responses.
`pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")` handles tokenisation and image encoding internally; pass the image as a PIL object alongside the text prompt. Multimodal models are larger and slower than text-only equivalents at the same parameter count — use quantization to fit them on a single GPU.
```python
from transformers import pipeline
from PIL import Image
pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")
image = Image.open("photo.jpg")
result = pipe({"image": image, "text": "Describe what you see."})
```

**Fine-tuning vs RAG** — Decision framework: fine-tuning for style/vocabulary consistency; RAG for frequently updated or private knowledge.
Fine-tuning embeds knowledge into weights — changes require retraining, but inference is faster and the model behaves consistently without retrieval latency. RAG keeps knowledge external and updatable at any time by updating the vector store, but adds a retrieval step and depends on chunk quality.
```python
# RAG — add new documents without retraining
index.add_documents(new_docs)

# Fine-tuning — requires full training pipeline; use for style, not facts
# trainer.train()  # see M5_2_FineTuning.ipynb
```

### M5_2_Dataset_prepare.ipynb

**Telegram parsing** — Extracts post text from a Telegram JSON export for use as fine-tuning data.
Telegram's `result.json` contains a list of messages; each has a `text` field that may be a plain string or a list of fragments (for posts with inline links). Filter for `type == "message"` and minimum text length to discard system events and very short posts. The non-obvious issue: check `isinstance(text, list)` and join fragments before processing.
```python
import json
with open("result.json") as f:
    data = json.load(f)
posts = [
    "".join(t if isinstance(t, str) else t["text"] for t in msg["text"])
    for msg in data["messages"]
    if msg.get("type") == "message" and len(str(msg.get("text", ""))) > 100
]
```

**LLM labeling** — Uses an LLM as an automated annotator to assign topic labels to raw text.
A classification prompt is sent for each sample; the model's response is used as the ground-truth label, reducing days of manual work to minutes. The technique introduces noise — verify a sample of labels manually. Use `chain.batch()` with `max_concurrency` to parallelize across the dataset.
```python
labels = label_chain.batch(
    [{"text": post} for post in posts],
    config={"max_concurrency": 5})
```

**Alpaca format** — Standard instruction fine-tuning template with Instruction, Input, and Response sections plus an EOS token.
Each training sample is formatted as `"### Instruction:\n{task}\n\n### Input:\n{context}\n\n### Response:\n{answer}{eos}"`. The EOS token is critical — without it the model learns to generate endlessly past the answer boundary during inference. `Input` can be empty for tasks without additional context.
```python
EOS = tokenizer.eos_token
def to_alpaca(row):
    return (f"### Instruction:\n{row['instruction']}\n\n"
            f"### Input:\n{row['input']}\n\n"
            f"### Response:\n{row['output']}{EOS}")
dataset = dataset.map(lambda x: {"text": to_alpaca(x)})
```

**HuggingFace upload** — Pushes a local dataset to HuggingFace Hub for reuse across training runs and sharing.
`dataset.push_to_hub("username/dataset-name", token="hf_...")` uploads all splits and creates a dataset card; the dataset is then loadable anywhere with `load_dataset("username/dataset-name")`. Keep the dataset private (`private=True`) if it contains personal data.
```python
from datasets import Dataset
hf_dataset = Dataset.from_pandas(df)
hf_dataset.push_to_hub("myuser/telegram-topics", token="hf_...", private=True)
```

### M5_2_FineTuning.ipynb

**Unsloth load** — Loads a quantized base model with Unsloth's patched kernels for faster training.
`FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.1-8B", load_in_4bit=True, max_seq_length=2048)` returns the model and tokenizer; Unsloth patches attention and MLP kernels at load time, giving 2–5× training speedup over stock HuggingFace. Only specific model architectures are supported — check Unsloth's model list before loading an arbitrary checkpoint.
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=2048, load_in_4bit=True)
```

**LoRA adapters** — Small trainable rank-decomposition matrices added on top of frozen base model weights.
`FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "v_proj", ...], lora_alpha=16)` attaches adapters to the specified attention projections; only these ~1–2% of parameters are updated. Higher `r` = more parameters updated = better fit but slower training and larger adapter files.
```python
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05)
```

**SFTTrainer** — TRL's supervised fine-tuning trainer; accepts a formatted text column and wraps HuggingFace TrainingArguments.
`SFTTrainer(model, train_dataset=dataset, dataset_text_field="text", ...)` handles batching, gradient accumulation, and LR scheduling. Key args: `per_device_train_batch_size`, `gradient_accumulation_steps` (effective batch = batch × accum), `learning_rate`, and `max_steps` (total gradient updates).
```python
from trl import SFTTrainer
from transformers import TrainingArguments
trainer = SFTTrainer(
    model=model, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4, max_steps=100, output_dir="outputs"))
trainer.train()
```

**Save adapters** — Persist only the LoRA adapter weights (~168 MB) rather than the full model (~16 GB).
`model.save_pretrained("lora_model")` saves adapter weights and config; `model.push_to_hub(...)` uploads to HuggingFace. At inference time, load the base model and apply the adapter with `PeftModel.from_pretrained(base_model, "lora_model")` — the base model is not duplicated.
```python
model.save_pretrained("lora_adapters")     # saves ~168 MB, not 16 GB
tokenizer.save_pretrained("lora_adapters")
# Load at inference time
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", ...)
model = PeftModel.from_pretrained(base, "lora_adapters")
```

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
