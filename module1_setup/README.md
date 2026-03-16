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
        {"role": "user", "content": "1+1"},
    ],
    temperature=0,
    max_tokens=100,
)
```

### Response Structure
```python
response.choices[0].message.content       # The generated text
response.usage.prompt_tokens              # Tokens in the input
response.usage.completion_tokens          # Tokens in the output
response.usage.total_tokens               # Sum
response.choices[0].finish_reason         # "stop" (normal) or "length" (cut off)
```

The course proxy also returns `available_tokens` — how many tokens you have left on your course key.

### Key Parameters
| Parameter | Range | Meaning |
|-----------|-------|---------|
| `temperature` | 0.0–2.0 | 0 = deterministic; higher = more random (default 1) |
| `max_tokens` | 1–context | Limit output length (default 256) |
| `top_p` | 0.0–1.0 | Nucleus sampling — use **either** `temperature` or `top_p`, not both |
| `n` | ≥1 | Number of completions to generate (default 1) |

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
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0.0, course_api_key=course_api_key)

template = """Вопрос: {question}
Ответ: Дай короткий ответ"""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = prompt | llm  # LCEL pipe syntax

print(chain.invoke("Когда человек первый раз полетел в космос?").content)
```

### 3. Open-Source Model (Remote via HuggingFace)
```python
import os
from getpass import getpass
from langchain_huggingface import HuggingFaceEndpoint

os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass("HuggingFace API key: ")

hf_llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
)

print(hf_llm.invoke("When did man first fly into space?"))
```

> Free HuggingFace API has a 250-token context window limit.

### 4. Open-Source Model (Local, No Internet)
```python
from langchain.llms import HuggingFacePipeline

bloom = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={"temperature": 1, "max_length": 64},
    # device=0,  # Uncomment if you have a GPU
)

print(bloom.invoke("When did man first fly into space?"))
```

> Small models (like Bloom 1B) give low-quality answers — the notebook shows it doesn't even know about Gagarin! Use for learning, not production.

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
- **Context Window**: The maximum number of tokens the model can process in one request (e.g., ~4097 for `gpt-3.5-turbo`).
- **Finish Reason**: `stop` = normal completion; `length` = output was cut off (increase `max_tokens`).
- **Temperature Trade-off**: Low temperature (0.0) for facts; high (1.0+) for creativity.

## Troubleshooting

- **"Invalid API key"** → Check that your course API key is correct and not expired.
- **Model not responding** → Check the course server status or switch to `gpt-3.5-turbo`.
- **Local model slow** → The model is downloading (~1–10 GB depending on size). Be patient or use the remote proxy instead.
