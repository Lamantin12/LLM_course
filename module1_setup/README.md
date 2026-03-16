# Module 1: API Setup & Model Access

Learn how to call Large Language Models through three different interfaces: the raw OpenAI API, LangChain's wrapper, and open-source models via HuggingFace.

## What You'll Learn

- How to call the OpenAI Chat Completions API directly
- Using the course proxy (`NDTOpenAI`) for free model access
- LangChain's `ChatOpenAI` abstraction layer
- Running open-source models remotely (HuggingFace)
- Running models locally without internet (`HuggingFacePipeline`)
- Parametric vs. source knowledge: what models know vs. what you tell them

## OpenAI API Anatomy

### Request Structure
```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ],
    temperature=0.7,
    max_tokens=100,
    top_p=0.9,
    n=1
)
```

### Response Structure
```python
response.choices[0].message.content       # The generated text
response.usage.prompt_tokens              # Tokens in the input
response.usage.completion_tokens          # Tokens in the output
response.usage.total_tokens               # Sum
response.finish_reason                    # "stop" (normal) or "length" (cut off)
```

### Key Parameters
| Parameter | Range | Meaning |
|-----------|-------|---------|
| `temperature` | 0.0–2.0 | 0 = deterministic; higher = more random |
| `max_tokens` | 1–context | Limit output length |
| `top_p` | 0.0–1.0 | Nucleus sampling; usually 0.9 |
| `n` | ≥1 | Number of completions to generate |

## Three Ways to Call a Model

### 1. Raw OpenAI Client (Direct API)
```python
from utils import NDTOpenAI

client = NDTOpenAI(api_key="your-course-key")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### 2. LangChain ChatOpenAI (Recommended)
```python
from utils import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key="your-course-key",
    temperature=0.7
)

response = llm.invoke("What is Python?")
print(response.content)
```

### 3. Open-Source Model (Remote via HuggingFace)
```python
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    huggingfacehub_api_token="your-hf-token"
)

response = llm.invoke("What is AI?")
print(response)
```

### 4. Open-Source Model (Local, No Internet)
```python
from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 50}
)

response = llm.invoke("What is coding?")
print(response)
```

## Parametric vs. Source Knowledge

**Parametric Knowledge** — stored in the model's weights during training
- Example: "Paris is the capital of France" (the model learned this from training data)
- Static; cannot be updated without retraining

**Source Knowledge** — injected into the prompt at runtime
- Example: Passing a paragraph about the latest stock prices in the prompt
- Dynamic; you control what information the model sees
- The model will only use information you explicitly provide

**Trade-off**: Larger models (like GPT-4) have more parametric knowledge and fewer hallucinations when using source knowledge.

## Notebooks in This Module

| Notebook | Topics |
|----------|--------|
| `M1_Welcome.ipynb` | Raw OpenAI client, LangChain ChatOpenAI, HuggingFace remote, local LLMs, knowledge types |

## Key Concepts

- **Tokens**: Units of text the model processes. ~4 characters ≈ 1 token.
- **Context Window**: The maximum number of tokens the model can process in one request (e.g., 4K, 16K, 128K).
- **Finish Reason**: `stop` = normal completion; `length` = output was cut off (increase `max_tokens`).
- **Temperature Trade-off**: Low temperature (0.0) for facts; high (1.0+) for creativity.

## Troubleshooting

- **"Invalid API key"** → Check that your course API key is correct and not expired.
- **Model not responding** → Check the course server status or switch to `gpt-3.5-turbo`.
- **Local model slow** → The model is downloading (~1–10 GB depending on size). Be patient or use the remote proxy instead.
