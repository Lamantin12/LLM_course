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
