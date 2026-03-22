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

## Lecture Notes

### M4_Agents.ipynb

**Tool Calling Agent** — Uses the LLM's native function-calling API to select tools, producing structured JSON instead of free-form text.
`create_tool_calling_agent(llm, tools, prompt)` + `AgentExecutor(agent=agent, tools=tools)` sets up the loop; the model outputs a JSON `{"name": "tool_name", "arguments": {...}}` object, which the executor parses and dispatches. More reliable than text-based ReAct because JSON output is validated, not regex-parsed — but requires a model that supports function calling (e.g. GPT-3.5-turbo, GPT-4).
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
executor.invoke({"input": "What is the weather in Berlin?"})
```

**RunnableWithMessageHistory** — Wraps any LCEL chain to add session-aware conversation memory without manual history management.
`RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input", history_messages_key="history")` intercepts each call, loads history for the `session_id`, prepends it, and writes the new turn back after the call. The `get_session_history` function maps a `session_id` string to a `BaseChatMessageHistory` store; each unique session ID gets an independent history.
```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
store = {}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
chain_with_history = RunnableWithMessageHistory(
    chain, get_history, input_messages_key="input", history_messages_key="history")
chain_with_history.invoke({"input": "Hi"}, config={"configurable": {"session_id": "u1"}})
```

**ReAct** — Extends Module 3's introduction with production details: error recovery and iteration limits.
`initialize_agent(..., handle_parsing_errors=True, max_iterations=10)` prevents two failure modes: crashing on malformed LLM output and running forever when stuck. Without `handle_parsing_errors`, a single bad response from the model crashes the entire executor; with it, the agent re-prompts itself.
```python
from langchain.agents import initialize_agent, AgentType
agent = initialize_agent(
    tools, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=10,
    verbose=True
)
```

**SQL Agent** — Translates natural-language questions to SQL and executes them against a live database.
`create_sql_agent(llm, db=SQLDatabase.from_uri("postgresql://..."), verbose=True)` inspects the schema automatically — table names, column types, sample rows — before writing a query. The non-obvious risk: the agent may run expensive or destructive queries; restrict it to a read-only database user and set `max_iterations`.
```python
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("postgresql://user:pw@localhost/dvdrental")
agent = create_sql_agent(llm, db=db, verbose=True)
agent.invoke("How many customers rented more than 5 films?")
```

**RAG Agent** — Gives an agent document-lookup capability by wrapping a retriever as a standard tool.
`create_retriever_tool(retriever, name="search_docs", description="...")` converts any LangChain retriever into a tool the agent can call by name; the description is what the LLM reads to decide when to use it — write it precisely. The agent can combine this tool with web search or SQL in one session, dynamically choosing which to query.
```python
from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    name="search_docs",
    description="Search the Gannibal biography for relevant information.")
agent = create_tool_calling_agent(llm, [retriever_tool, ...], prompt)
```

**LangServe** — Deploys any LangChain runnable as a REST API with auto-generated FastAPI endpoints and a playground UI.
`add_routes(app, chain, path="/chat")` generates `/chat/invoke`, `/chat/stream`, and `/chat/batch` endpoints; visiting `/chat/playground` opens an interactive browser UI — zero additional code required. The non-obvious requirement: the chain must be serializable (no lambdas or local closures); use `RunnableLambda` or named LCEL components.
```python
from fastapi import FastAPI
from langserve import add_routes
app = FastAPI()
add_routes(app, chain, path="/chat")
# uvicorn app:app --port 8000
# → /chat/invoke, /chat/stream, /chat/playground
```

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
