# Module 4: LangChain Agents

Build autonomous LLM-powered agents that choose which tools to call, chain reasoning steps, query databases, and retrieve documents — all without hardcoded control flow.

## What You'll Learn

- Agent architecture: tools, prompts, agent scratchpad, and the agent executor loop
- Tool Calling Agent: the simplest agent type with automatic tool selection
- Conversation memory: session-based chat history with `RunnableWithMessageHistory`
- ReAct (Reasoning + Acting): step-by-step reasoning with external tool use
- Structured Chat Agent: handling tools with multiple parameters
- Self-Ask with Search: decomposing complex questions into sub-questions
- Agent + RAG: combining retrieval-augmented generation with agent reasoning
- Agent + SQL: natural-language queries translated to SQL automatically
- Hierarchical agents: orchestrating multiple sub-agents (SQL + Python)
- LangServe: deploying agents as REST APIs with FastAPI

## Agent Types Overview

```
┌──────────────────────┐
│  Tool Calling Agent  │  Simplest. LLM decides which tool to call via function calling.
├──────────────────────┤
│  ReAct Agent         │  Thought → Action → Observation loop. Single-param tools only.
├──────────────────────┤
│  Structured Chat     │  Like ReAct but supports multi-parameter tools.
├──────────────────────┤
│  Self-Ask + Search   │  Breaks complex questions into sub-questions answered by search.
└──────────────────────┘
```

---

## Tool Calling Agent

The LLM automatically selects which tool to invoke based on the user query:

```python
from langchain_classic.agents import load_tools, tool, create_tool_calling_agent, AgentExecutor

tools = load_tools(["llm-math"], llm=llm)

@tool
def get_word_length(word: str) -> int:
    """Возвращает длину слова"""
    return len(word)

tools.append(get_word_length)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты полезный ассистент"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # required — stores intermediate steps
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
executor.invoke({"input": "Сколько букв в слове зачёт?"})
```

## Agent with Memory

Add `chat_history` to the prompt and wrap the executor with `RunnableWithMessageHistory`:

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

agent_with_history = RunnableWithMessageHistory(
    executor, get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
agent_with_history.invoke(
    {"input": "А в слове ёж?"},
    config={"configurable": {"session_id": "1"}},
)
```

## ReAct Agent

Implements the Thought → Action → Observation loop using prompts from LangChain Hub:

```python
from langchain_classic import hub
from langchain_classic.agents import create_react_agent

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

**Limitation**: ReAct agents don't support multi-parameter tools — use Structured Chat Agent instead.

## Structured Chat Agent

Handles tools with multiple input parameters (e.g., triangle area with sides a, b, c):

```python
from langchain_classic.agents import create_structured_chat_agent

prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(llm, tools, prompt)
```

## Self-Ask with Search

Decomposes complex multi-hop questions into intermediate sub-questions:

```python
from langchain_classic.agents import create_self_ask_with_search_agent, Tool
from langchain_community.utilities import SerpAPIWrapper

search = SerpAPIWrapper()
tools = [Tool(name="Intermediate Answer", func=search.run, description="...")]
agent = create_self_ask_with_search_agent(llm, tools, prompt)
```

## Agent + RAG

Load a web page, embed it into a FAISS vector store, and give the agent a retriever tool:

```python
from langchain_classic.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(retriever, "search_web", "Searches and returns data")
agent = create_react_agent(llm, [retriever_tool], prompt)
```

## Agent + SQL Database

Automatically translates natural-language questions into SQL queries:

```python
from langchain_community.agent_toolkits import create_sql_agent
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit

db = SQLDatabase.from_uri("sqlite:///data/pizzeria.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm, toolkit=toolkit, verbose=True)
agent.invoke({"input": "В каком месяце 2023 года было больше всего заказов?"})
```

## Super Agent (Hierarchical)

Combine SQL and Python sub-agents into a top-level orchestrator:

```python
tools = [
    Tool(name="PythonAgent", func=python_agent.run, description="Run python commands"),
    Tool(name="SQLAgent", func=sql_agent.run, description="Query sql tables"),
]
super_agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)
```

## LangServe Deployment

Deploy agents as REST endpoints using FastAPI + LangServe:

```python
from langserve import add_routes

app = FastAPI()
add_routes(app, agent_executor, path="/rag_agent")
# Run: uvicorn langserve_app:app --port 8501
```

---

## Files in This Module

| File | Description |
|------|-------------|
| `M4_Agents.ipynb` | Lecture notebook — all agent types with interactive examples |
| `tool_calling_agent.py` | Tool Calling Agent + Agent with Memory |
| `react_agent.py` | ReAct Agent + Structured Chat Agent |
| `self_ask_agent.py` | Self-Ask with Search Agent |
| `rag_agent.py` | Agent + RAG (web page retrieval) |
| `sql_agent.py` | SQL Agent + Super Agent (hierarchical) |
| `langserve_app.py` | LangServe FastAPI deployment server |

## Setup

1. Install dependencies:
   ```bash
   pip install langchain langchain-community langchain-openai langchain-classic langchainhub \
       langchain-experimental langserve[all] faiss-cpu google-search-results numexpr
   ```

2. Set environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_course_api_key
   SERPAPI_API_KEY=your_serpapi_key  # optional, needed for search-based agents
   ```

3. Run any script:
   ```bash
   python module4_agents/tool_calling_agent.py
   python module4_agents/sql_agent.py
   ```

## Key Concepts

- **Agent Scratchpad**: A prompt placeholder that stores intermediate reasoning and tool results during execution
- **Toolkits**: Pre-built collections of related tools (e.g., `SQLDatabaseToolkit`)
- **LangChain Hub**: Central repository for curated prompts (`hub.pull("hwchase17/react")`)
- **`handle_parsing_errors`**: Lets the agent recover from malformed LLM output instead of crashing
- **`max_iterations`**: Safety limit on how many Thought→Action→Observation loops the agent can run
