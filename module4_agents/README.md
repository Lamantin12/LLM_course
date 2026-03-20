# Module 4 — LangChain Agents

Build autonomous LLM-powered agents that choose which tools to call, chain reasoning steps, query databases, and retrieve documents — all without hardcoded control flow.

---

## Topics Covered

- Agent architecture — tools, prompt, agent scratchpad, executor loop
- Tool Calling Agent — simplest type with automatic tool selection via function calling
- Conversation memory — session-based chat history with `RunnableWithMessageHistory`
- ReAct agent — Thought → Action → Observation loop
- Structured Chat Agent — multi-parameter tools
- Self-Ask with Search — decomposing complex questions into sub-questions
- Agent + RAG — combining retrieval with agent reasoning
- Agent + SQL — natural-language queries translated to SQL automatically
- Hierarchical agents — orchestrating multiple sub-agents
- LangServe — deploying agents as REST APIs with FastAPI

---

## Files

| File | Description |
|------|-------------|
| `M4_Agents.ipynb` | Lecture notebook — all agent types with interactive examples |
| `M4_2_exercises.ipynb` | Exercise notebook |
| `langserve_app.py` | LangServe FastAPI deployment server |
| `task_1_gannibal_rag_agent.py` | **Exercise Task 1** — Gannibal RAG agent with FAISS vector store |
| `task_2_dvdrental_sql_agent.py` | **Exercise Task 2** — SQL Agent on PostgreSQL DVD rental database |
| `task_3_polygraph_agent.py` | **Exercise Task 3** — True/False fact-checker agent |
| `gannibal_faiss_index/` | Persisted FAISS index for the Gannibal RAG agent |

---

## How to Run

```bash
pip install langchain langchain-community langchain-openai langchain-classic langchainhub \
    langchain-experimental langserve[all] faiss-cpu google-search-results numexpr

# Set environment variables
cp .env.example .env  # add OPENAI_API_KEY and optionally SERPAPI_API_KEY

python module4_agents/task_1_gannibal_rag_agent.py
python module4_agents/task_2_dvdrental_sql_agent.py
python module4_agents/task_3_polygraph_agent.py

# LangServe API server
uvicorn module4_agents.langserve_app:app --port 8501
```

---

## Key Concepts

- **Agent Scratchpad**: Prompt placeholder that accumulates intermediate reasoning steps and tool results during execution.
- **Tool Calling Agent**: Uses the LLM's native function-calling API to select and invoke tools — simpler and more reliable than ReAct for supported models.
- **ReAct loop**: Thought → Action → Observation — the agent reasons in text, then acts, then observes the result; repeats until it has a final answer.
- **Structured Chat Agent**: Extends ReAct to support tools with multiple input parameters.
- **`handle_parsing_errors`**: Lets the agent recover from malformed LLM output instead of raising an exception.
- **`max_iterations`**: Safety limit on Thought→Action→Observation loops — always set to prevent runaway agents.
- **`create_retriever_tool`**: Wraps a LangChain retriever as a standard agent tool — bridges RAG and agents.
- **LangServe**: FastAPI extension that auto-generates REST endpoints and a playground UI from a LangChain runnable.
