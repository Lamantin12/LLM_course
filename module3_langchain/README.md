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

## Lecture Notes

### M3_LangChain_Chains.ipynb

**LLMChain** — Legacy wrapper that binds a prompt template and an LLM into a single callable with a named output key.
`LLMChain(llm=llm, prompt=prompt, output_key="review")` names the output so downstream chains can reference it by variable name. It is deprecated in favour of LCEL but widely seen in existing code; the key conceptual contribution is making chains composable by naming their outputs.
```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")
result = chain({"product": "smartwatch"})
print(result["review"])
```

**TransformChain** — Inserts a pure Python function into a pipeline without making an LLM call.
`TransformChain(input_variables=["text"], output_variables=["clean_text"], transform=fn)` passes data through any callable — useful for regex cleaning, JSON parsing, or format conversion between chain steps. The critical design rule: the function must accept and return a `dict`; missing keys cause a `KeyError` inside the chain.
```python
from langchain.chains import TransformChain
import re
def clean(inputs):
    return {"clean_text": re.sub(r"\s+", " ", inputs["text"]).strip()}
transform_chain = TransformChain(
    input_variables=["text"], output_variables=["clean_text"], transform=clean)
```

**SequentialChain** — Wires multiple chains together by matching output keys of one to input variables of the next.
`SequentialChain(chains=[c1, c2], input_variables=["product"], output_variables=["review", "summary"])` runs chains left to right; each chain's `output_key` becomes available to all later chains. The non-obvious gotcha: all intermediate output keys must be globally unique — duplicate names cause silent overwrites.
```python
from langchain.chains import SequentialChain
seq = SequentialChain(
    chains=[review_chain, summary_chain],
    input_variables=["product"],
    output_variables=["review", "summary"])
result = seq({"product": "laptop"})
```

**LCEL** — Modern declarative chain syntax using the `|` pipe operator; the recommended replacement for all legacy chain classes.
`chain = prompt | llm | StrOutputParser()` composes runnables left to right; every segment implements the `Runnable` interface (`.invoke`, `.stream`, `.batch`). `RunnablePassthrough()` passes an input unchanged, used to thread a value alongside a transformed one in parallel dict composition.
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
print(chain.invoke("What is LCEL?"))
```

**RunnableBranch** — Conditional routing: evaluates predicate functions in order and runs the first matching sub-chain.
`RunnableBranch([(condition_fn, chain_a), (condition_fn2, chain_b)], default_chain)` reads like a `switch`/`match` — the first predicate returning `True` wins. Combine with an LLM classifier: classify the query into a category string, then branch on that string. The non-obvious detail: predicates receive the full input dict, so classifier output must be stored under a key the predicate can access.
```python
from langchain_core.runnables import RunnableBranch
branch = RunnableBranch(
    (lambda x: x["topic"] == "botany", botany_chain),
    (lambda x: x["topic"] == "football", football_chain),
    general_chain   # default
)
```

### M3_LangChain_Agents_intro.ipynb

**@tool decorator** — Converts a plain Python function into a LangChain tool the agent can call.
The decorator uses the function name as the tool name and the docstring as the description the LLM reads to decide whether to use it — write concise, unambiguous docstrings. The non-obvious rule: if an agent has only one narrow tool and receives an off-topic query, it raises a `ValueError` — always provide a general fallback tool.
```python
from langchain.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word."""
    return len(word)
```

**ReAct** — Reasoning + Acting loop where the agent alternates between Thought, Action, and Observation steps.
Each iteration: the LLM writes a `Thought`, outputs an `Action` (tool name + input), the executor runs the tool and appends the `Observation`, then repeats until done. Set `verbose=True` to see the full loop — essential for debugging. Always set `max_iterations` to prevent infinite loops.
```python
from langchain.agents import initialize_agent, AgentType
agent = initialize_agent(
    tools, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, max_iterations=5, handle_parsing_errors=True
)
agent.run("How many letters are in 'зачёт'?")
```

**Built-in tools** — Pre-packaged LangChain tools for Wikipedia, DuckDuckGo, Arxiv, and Python execution.
`WikipediaQueryRun`, `DuckDuckGoSearchRun`, `ArxivQueryRun`, and `PythonREPLTool` all implement the `Runnable` interface and can be dropped into any agent without writing a `@tool` function. The agent dynamically selects which to call based on the query — unlike `RouterChain`, the order is not hardcoded. Only use `PythonREPLTool` in sandboxed environments.
```python
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
tools = [
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    DuckDuckGoSearchRun()
]
```

### M3_LangChain_Memory.ipynb

**BufferMemory** — Stores the full conversation transcript verbatim in a `{history}` prompt variable.
`ConversationBufferMemory(memory_key="history", return_messages=True)` passes every prior turn to the model on each call, so it can refer to anything said earlier. The critical limitation: history grows unbounded — for long conversations this quickly exceeds the context window and increases cost per call.
```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
chain = ConversationChain(llm=llm, memory=memory)
chain.predict(input="My name is Alex")
chain.predict(input="What's my name?")  # "Your name is Alex"
```

**WindowMemory** — Keeps only the last `k` conversation turns, discarding older history.
`ConversationBufferWindowMemory(k=1)` stores only the single most recent exchange; older turns are silently dropped. This bounds memory cost regardless of conversation length, but the model immediately forgets context older than `k` turns — the demo shows it forgetting a name after just one exchange.
```python
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=2)  # remember last 2 turns
```

**TokenBufferMemory** — Trims conversation history by token count instead of turn count.
`ConversationTokenBufferMemory(llm=llm, max_token_limit=50)` uses the model's tokenizer to measure history length and drops oldest turns when the limit is reached. Passing `llm=llm` is required so the correct tokenizer is used — mismatched tokenizers give inaccurate counts.
```python
from langchain.memory import ConversationTokenBufferMemory
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=100)
```

**SummaryMemory** — Compresses conversation history into a rolling plain-text summary before each call.
`ConversationSummaryMemory(llm=llm)` makes a secondary LLM call after each turn to update the summary using `{summary}` + `{new_lines}` → new summary. The non-obvious artefact: the internal summarisation prompt is English-only, so the summary switches to English regardless of the original conversation language.
```python
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)
chain = ConversationChain(llm=llm, memory=memory, verbose=True)
```

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
