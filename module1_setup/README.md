# Module 1 — API Setup & Model Access

Learn how to call Large Language Models through three different interfaces: the raw OpenAI API, LangChain's wrapper, and open-source models via HuggingFace.

---

## Topics Covered

- Raw OpenAI Chat Completions API — direct calls, request/response anatomy
- Course proxy (`NDTOpenAI`) — free model access via `utils.py`
- LangChain `ChatOpenAI` — abstraction layer with LCEL pipe syntax
- HuggingFace remote endpoints — open-source models without local GPU
- HuggingFace local pipeline — running models without internet
- Parametric vs. source knowledge — what models know vs. what you tell them

---

## Lecture Notes

### M1_Welcome.ipynb

**Raw OpenAI API** — Direct access to the Chat Completions endpoint, showing the full request/response anatomy.
The client sends messages as a list of role/content dicts and receives a `ChatCompletion` object; key fields are `choices[0].message.content`, `finish_reason` (`stop` vs `length`), and `usage.total_tokens`. When `finish_reason == "length"`, the output was cut off — increase `max_tokens` to fix it.
```python
from openai import OpenAI
client = OpenAI(base_url="https://api.neuraldeep.tech/", api_key="...")
resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
print(resp.choices[0].message.content, resp.usage.total_tokens)
```

**NDTOpenAI proxy** — Course wrapper that routes requests through the course server instead of OpenAI directly.
`NDTOpenAI` from `utils.py` is a drop-in replacement for `OpenAI()`; the interface is identical but adds an `available_tokens` field showing your remaining API budget. Use it instead of the raw client throughout the course to avoid spending personal quota.
```python
from utils import NDTOpenAI
client = NDTOpenAI()
resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=[...])
print(resp.available_tokens)   # course-specific extension
```

**ChatOpenAI pipe** — LangChain abstraction that wraps the model and slots into LCEL chains with the `|` operator.
`chain = prompt | llm` collapses boilerplate request/response handling into one expression; `.invoke({"q": ...})` returns an `AIMessage`. Add `StrOutputParser()` as a third pipe segment to unwrap the message to a plain string, keeping the chain composable with downstream steps.
```python
from utils import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI()
chain = ChatPromptTemplate.from_template("Answer: {q}") | llm | StrOutputParser()
print(chain.invoke({"q": "What is 2+2?"}))
```

**HuggingFace remote** — Calls open-source models hosted on the HuggingFace Inference API without a local GPU.
`HuggingFaceEndpoint(repo_id="...", huggingfacehub_api_token="hf_...")` sends requests to HuggingFace servers; the free tier caps context at 250 tokens and adds network latency. This is the lowest-friction way to test open-source models but is rate-limited.
```python
from langchain_community.llms import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token="hf_..."
)
print(llm.invoke("What is the capital of France?"))
```

**HuggingFace local** — Runs a model entirely on your machine after a one-time download, with no internet required.
`HuggingFacePipeline.from_model_id("bigscience/bloom-1b7", task="text-generation")` downloads and caches weights, then runs inference locally. The key gotcha: small local models like `bloom-1b7` produce factually wrong answers — the notebook shows it attributing the first human spaceflight to John Glenn, illustrating that parametric recall degrades with model size.
```python
from langchain_community.llms import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100}
)
print(llm.invoke("Who was the first human in space?"))
```

**Parametric vs source knowledge** — The distinction between what the model learned during training and what you inject at runtime.
Parametric knowledge is baked into model weights — static, potentially stale, and weaker in smaller models. Source knowledge is text you place in the prompt at inference time; it overrides parametric knowledge and is the foundation of RAG. The bloom-1b7 hallucination is the concrete motivation for always injecting source context when accuracy matters.
```python
# Parametric only — model may hallucinate
llm.invoke("Who invented the telephone?")

# Source knowledge injected — grounded answer
context = "Bell's 1876 patent credited Alexander Graham Bell with inventing the telephone."
llm.invoke(f"Context: {context}\n\nWho invented the telephone?")
```

---

## Files

| File | Description |
|------|-------------|
| `M1_Welcome.ipynb` | Raw OpenAI client, LangChain ChatOpenAI, HuggingFace remote and local LLMs, knowledge types |

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook module1_setup/M1_Welcome.ipynb
```

Set your course API key when prompted in the notebook.

---

## Key Concepts

- **Tokens**: Units of text the model processes (~4 characters ≈ 1 token).
- **Context Window**: Maximum tokens the model can process in one request (e.g., ~4097 for `gpt-3.5-turbo`).
- **Temperature**: Controls output randomness. 0.0 = deterministic; 1.0+ = creative.
- **`finish_reason`**: `stop` = normal completion; `length` = output cut off (increase `max_tokens`).
- **Parametric Knowledge**: Information stored in the model's weights during training — static, cannot be updated without retraining.
- **Source Knowledge**: Information injected into the prompt at runtime — dynamic, you control it.
- **`top_p`**: Nucleus sampling; use either `temperature` or `top_p`, not both simultaneously.
- **`available_tokens`**: Course proxy extension — shows remaining token budget for your API key.
