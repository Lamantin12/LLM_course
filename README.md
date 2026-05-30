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
import os
from langchain_openai import ChatOpenAI

BASE_URL = "https://api.vsellm.ru/"
API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-4o-mini",
    base_url=BASE_URL,
    temperature=0.0,
)

response = llm.invoke("What is the capital of France?")
print(response.content)
```

## Modules

| Module | Directory | Topic | Key Libraries | Main Notebook(s) |
|--------|-----------|-------|---------------|-----------------|
| **1** | [module1_setup/](module1_setup/README.md) | API Setup & Model Access | `openai`, `langchain_openai`, `transformers` | `M1_Welcome.ipynb` |
| **2** | [module2_prompt_engineering/](module2_prompt_engineering/README.md) | Prompt Engineering | `langchain.prompts`, `langchain.output_parsers` | `M2_1_*`, `M2_2_*` |
| **3** | [module3_langchain/](module3_langchain/README.md) | LangChain Framework | `langchain.chains`, `langchain.agents`, `langchain.memory` | `M3_*` |
| **4** | [module4_rag/](module4_rag/README.md) | Retrieval-Augmented Generation | `langchain.document_loaders`, `langchain.vectorstores` | `M4_RAG.ipynb` |
| **4** | [module4_advanced_prompt_engineering/](module4_advanced_prompt_engineering/README.md) | Advanced Prompt Engineering | `langchain_experimental` | `M4_Advanced_Prompting.ipynb` |
| **4** | [module4_agents/](module4_agents/README.md) | Agents | `langchain.agents`, `langserve` | `M4_Agents.ipynb` |
| **5** | [module5_fine_tuning/](module5_fine_tuning/README.md) | Open-Source Models & Fine-tuning | `transformers`, `langchain_community`, `torch`, `bitsandbytes`, `unsloth`, `trl`, `peft` | `M5_Zoo.ipynb`, `M5_2_Dataset_prepare.ipynb`, `M5_2_FineTuning.ipynb` |

## Directory Structure

```
LLM/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── utils.py                                     # Legacy course API wrappers (kept for older notebooks); new code uses langchain_openai.ChatOpenAI directly with base_url=https://api.vsellm.ru/
│
├── module1_setup/
│   ├── README.md
│   └── M1_Welcome.ipynb
│
├── module2_prompt_engineering/
│   ├── README.md
│   ├── M2_1_Prompt_Engineering_intro.ipynb
│   ├── M2_2_LangChain_Prompting.ipynb
│   ├── M2_1_exercises.ipynb
│   └── M2_2_exercises.ipynb
│
├── module3_langchain/
│   ├── README.md
│   ├── M3_LangChain_Chains.ipynb
│   ├── M3_LangChain_Agents_intro.ipynb
│   ├── M3_LangChain_Memory.ipynb
│   ├── M3_1_exercises.ipynb
│   ├── M3_2_exercises.ipynb
│   └── M3_3_exercises.ipynb
│
├── module4_rag/
│   ├── README.md
│   ├── M4_RAG.ipynb
│   ├── custom_text_splitter.py
│   ├── pushkin_rag.py
│   └── pushkin_questions_data/
│
├── module4_advanced_prompt_engineering/
│   ├── README.md
│   ├── M4_Advanced_Prompting.ipynb
│   ├── M4_3_exercises.ipynb
│   ├── advanced_prompting.py
│   ├── task_1_sudoku_tot.py
│   └── task_2_pal_math.py
│
├── module4_agents/
│   ├── README.md
│   ├── M4_Agents.ipynb
│   ├── M4_2_exercises.ipynb
│   ├── langserve_app.py
│   ├── task_1_gannibal_rag_agent.py
│   ├── task_2_dvdrental_sql_agent.py
│   ├── task_3_polygraph_agent.py
│   └── gannibal_faiss_index/
│
├── module5_fine_tuning/
│   ├── README.md
│   ├── M5_Zoo.ipynb
│   ├── M5_2_Dataset_prepare.ipynb
│   └── M5_2_FineTuning.ipynb
│
└── submissions/
    └── (CSV results from exercises, named m[module-id]_[section]_[exercise]_solution.csv)
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
