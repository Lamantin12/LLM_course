# LLM Practical Course

A hands-on course for building applications with Large Language Models using LangChain and the OpenAI API.

## Prerequisites

- Python 3.10+
- `pip` or another package manager
- A course API key (provided during enrollment)

## Setup

**Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```python
from utils import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.7,
    course_api_key="your-course-key"
)

response = llm.invoke("What is the capital of France?")
print(response.content)
```

## Modules

| Module | Topic | Key Libraries | Notebooks |
|--------|-------|---------------|-----------|
| **1** | API Setup & Model Access | `openai`, `langchain_openai`, `transformers` | `M1_Welcome.ipynb` |
| **2** | Prompt Engineering | `langchain.prompts`, `langchain.output_parsers` | `M2_1_*`, `M2_2_*` |
| **3** | LangChain Framework | `langchain.chains`, `langchain.agents`, `langchain.memory` | `M3_*` |
| **4** | Retrieval-Augmented Generation (RAG) | `langchain.document_loaders`, `langchain.vectorstores` | `M4_RAG.ipynb` |

## Directory Structure

```
LLM/
├── README.md                                    # This file
├── CLAUDE.md                                    # Project metadata
├── requirements.txt                             # Dependencies
├── utils.py                                     # Course API wrappers
│
├── module1_setup/
│   ├── README.md
│   └── M1_Welcome.ipynb
│
├── module2_prompt_engineering/
│   ├── README.md
│   ├── M2_1_Prompt_Engineering_intro.ipynb
│   ├── M2_2_LangChain_Prompting.ipynb
│   └── (solution notebooks and submissions)
│
├── module3_langchain/
│   ├── README.md
│   ├── M3_LangChain_Chains.ipynb
│   ├── M3_LangChain_Agents_intro.ipynb
│   ├── M3_LangChain_Memory.ipynb
│   └── (solution notebooks and submissions)
│
├── module4_rag/
│   ├── README.md
│   ├── M4_RAG.ipynb
│   ├── custom_text_splitter.py
│   ├── pushkin_rag.py
│   └── pushkin_questions_data/
│
└── submissions/
    └── (CSV results from exercises)
```

## Course API Reference

### Available Models
- `gpt-3.5-turbo` — fast, general-purpose model
- `gpt-4` — more capable, reasoning-heavy tasks (if available via your course)

### Key Parameters
- `temperature` (0.0–2.0): Controls randomness. 0.0 = deterministic, 1.0+ = creative
- `max_tokens`: Maximum tokens in the completion
- `top_p`: Nucleus sampling; use **either** `temperature` **or** `top_p`, not both simultaneously
- `n`: Number of completions to generate

## Learning Path

1. **Module 1** → Understand how to call an LLM and the difference between parametric (model weights) and source knowledge (injected context)
2. **Module 2** → Master prompting techniques: structure, temperature, few-shot examples, and output parsing
3. **Module 3** → Build applications: chains (sequential logic), agents (tool use), and memory (conversation context)
4. **Module 4** → Extend the model with external knowledge: load documents, chunk them, embed them, search them, and feed results into the LLM

## Resources

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [LangChain Documentation](https://python.langchain.com/)
- [HuggingFace Models](https://huggingface.co/models)

## Notes

- All notebooks are compatible with Google Colab and local Jupyter
- Notebooks are in Russian; code is in English
- Solutions to exercises are provided in separate `*_solution.ipynb` notebooks
