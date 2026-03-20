# Module 3 — LangChain Framework: Chains, Agents & Memory

Build sophisticated LLM applications by composing chains (sequential logic), agents (tool-using AI), and memory (conversation context) using LCEL.

---

## Topics Covered

- **Chains** — combine prompts, models, and Python logic in sequences
- **LCEL** — modern declarative syntax using the `|` pipe operator
- **TransformChain** — insert pure Python functions into a chain
- **SequentialChain** — multi-step pipelines passing outputs to inputs
- **RunnableBranch** — route queries to specialist chains based on classification
- **Agents** — LLM decides which tools to use and in what order (ReAct loop)
- **Tool creation** — `@tool` decorator and `Tool` class; docstrings matter
- **Built-in tools** — Wikipedia, DuckDuckGo, Arxiv, PythonREPL
- **Memory** — Buffer, Window, TokenBuffer, and Summary conversation memory

---

## Files

| File | Description |
|------|-------------|
| `M3_LangChain_Chains.ipynb` | LLMChain, TransformChain, SequentialChain, LCEL, RunnableBranch |
| `M3_LangChain_Agents_intro.ipynb` | Tools, `@tool` decorator, agents, ReAct loop, built-in tools |
| `M3_LangChain_Memory.ipynb` | ConversationChain, BufferMemory, WindowMemory, TokenBufferMemory, SummaryMemory |
| `M3_1_exercises.ipynb` | Exercises for section 3.1 (chains) |
| `M3_2_exercises.ipynb` | Exercises for section 3.2 (agents) |
| `M3_3_exercises.ipynb` | Exercises for section 3.3 (memory) |

---

## How to Run

```bash
pip install langchain langchain-community langchain-openai openai wikipedia duckduckgo-search arxiv
jupyter notebook module3_langchain/M3_LangChain_Chains.ipynb
```

---

## Key Concepts

- **LCEL pipe syntax**: `chain = prompt | llm | parser` — one-line chain definition that supports `.invoke()`, `.stream()`, and `.batch()`.
- **ReAct loop**: Thought → Action → Observation — the agent reasons about what to do, calls a tool, inspects the result, and repeats until done.
- **Tool docstring**: The LLM reads the tool's docstring to decide whether to use it — write clear, specific descriptions.
- **`handle_parsing_errors`**: Lets the agent recover from malformed LLM output instead of crashing; set `max_iterations` to prevent infinite loops.
- **BufferMemory**: Stores full conversation history — accurate but grows unbounded and expensive.
- **WindowMemory**: Keeps only the last `k` exchanges — bounded cost, forgets older context.
- **SummaryMemory**: Compresses history into a running summary — compact but introduces slight distortion.
- **`LLMMathChain` (deprecated)**: Replaced by agents with tools — do not use in new code.
