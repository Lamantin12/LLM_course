# Module 3: LangChain Framework — Chains, Agents & Memory

Build sophisticated LLM applications by composing chains (sequential logic), agents (tool-using AI), and memory (conversation context). Master LCEL (LangChain Expression Language) for declarative application design.

## What You'll Learn

- **Chains**: Combine prompts, models, and logic in sequences
- **LCEL**: Modern declarative syntax using the `|` pipe operator
- **Agents**: Let the LLM decide which tools to use and in what order
- **Memory**: Store conversation history so the LLM remembers context
- **Routing**: Direct queries to specialist chains based on content

## Chains: Sequential Logic

### LLMChain (Legacy, Now LCEL)

Connect a prompt + LLM into a single pipeline:

```python
from langchain import PromptTemplate, LLMChain
from utils import ChatOpenAI

prompt = PromptTemplate(template="What is {topic}?", input_variables=["topic"])
llm = ChatOpenAI()
chain = LLMChain(prompt=prompt, llm=llm)

result = chain.invoke({"topic": "machine learning"})
print(result["text"])
```

### LCEL (Modern Syntax)

LCEL is cleaner, more composable, and handles streaming natively:

```python
from langchain import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from utils import ChatOpenAI

prompt = PromptTemplate(template="What is {topic}?", input_variables=["topic"])
llm = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "machine learning"})
print(result)
```

**Benefit**: One-line chain definition vs. 10+ lines of setup code.

### TransformChain (Pure Python Functions)

Insert Python logic into a chain:

```python
from langchain.chains import TransformChain

def clean_text(text):
    return {"output": text.lower().strip()}

transform = TransformChain(
    input_variables=["input"],
    output_variables=["output"],
    transform=clean_text
)

result = transform.invoke({"input": "  HELLO WORLD  "})
print(result["output"])  # "hello world"
```

### SequentialChain (Multiple Steps)

Run chains in sequence, passing outputs to inputs:

```python
from langchain.chains import SequentialChain, LLMChain
from langchain import PromptTemplate
from utils import ChatOpenAI

llm = ChatOpenAI()

# Step 1: Rewrite in a style
prompt1 = PromptTemplate(template="Rewrite as a {style}:\n{text}", input_variables=["text", "style"])
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="styled_text")

# Step 2: Summarize
prompt2 = PromptTemplate(template="Summarize:\n{styled_text}", input_variables=["styled_text"])
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="summary")

full_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["text", "style"],
    output_variables=["styled_text", "summary"]
)

result = full_chain({"text": "The sky is blue", "style": "formal letter"})
print(result)
```

### Streaming (Token-by-Token Output)

Stream responses as they're generated:

```python
chain = prompt | llm | parser

for token in chain.stream({"topic": "AI"}):
    print(token, end="", flush=True)
```

### RunnableBranch (Router Chains)

Route queries to different chains based on classification:

```python
from langchain.schema.runnable import RunnableBranch

classifier_prompt = PromptTemplate(
    template="Classify as botany or football:\n{query}",
    input_variables=["query"]
)

botany_chain = PromptTemplate(template="Botany fact: {query}") | llm
football_chain = PromptTemplate(template="Football fact: {query}") | llm

router = RunnableBranch(
    (classifier_chain | parser | is_botany, botany_chain),
    (classifier_chain | parser | is_football, football_chain),
    default_chain=botany_chain
)

result = router.invoke({"query": "What is photosynthesis?"})
```

## Agents: Tool-Using AI

### What's an Agent?

An agent is an LLM that can:
1. Receive a query
2. Decide which tools to use
3. Execute those tools
4. Examine the results
5. Decide the next step (repeat or return answer)

This is the **ReAct loop** (Reasoning + Acting).

### Creating Tools

```python
from langchain.agents import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Or manually:
from langchain.agents import Tool

def subtract(a, b):
    return a - b

tools = [
    Tool(name="Add", func=add, description="Add two numbers"),
    Tool(name="Multiply", func=multiply, description="Multiply two numbers"),
    Tool(name="Subtract", func=subtract, description="Subtract b from a"),
]
```

**Important**: Write clear docstrings; the LLM reads them to decide which tool to use.

### Built-in Tools

```python
from langchain.agents import load_tools
from langchain_experimental.tools import PythonREPLTool

# Search tools
tools = load_tools(["wikipedia", "arxiv", "ddg-search"], llm=llm)

# Code execution
python_repl = PythonREPLTool()

# Math chains
from langchain.chains import LLMMathChain
math_chain = LLMMathChain.from_llm(llm)
```

### Initializing an Agent

```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True
)

result = agent.run("What is the current population of France?")
```

**Parameters**:
- `verbose=True`: Print reasoning steps
- `max_iterations`: Prevent infinite loops
- `handle_parsing_errors=True`: Gracefully handle LLM parsing failures

### When LLMs Fail

LLMs struggle with tasks that require exact computation or current information:

```python
# ❌ This will fail
llm.invoke("How many letters in 'photosynthesis'?")
# → Might say 15 (wrong; it's 14) because of tokenisation issues

# ✅ Use an agent with a tool
@tool
def count_letters(word: str) -> int:
    """Count letters in a word."""
    return len(word)

agent.run("How many letters in 'photosynthesis'?")
# → 14 (correct; used the tool)
```

## Memory: Conversation Context

### The Problem

LLMs are **stateless**. Each call is independent:

```python
llm.invoke("My name is Alice")
# → "Nice to meet you, Alice!"

llm.invoke("What is my name?")
# → "I don't know your name." (forgot!)
```

### The Solution: ConversationChain

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.run("My name is Alice")
conversation.run("What is my name?")
# → "Your name is Alice." (remembered!)
```

### Memory Types

| Type | Stores | Pros | Cons |
|------|--------|------|------|
| **BufferMemory** | Full history | Accurate | Grows unbounded; expensive |
| **WindowMemory** | Last `k` messages | Cheap; bounded | Forgets older context |
| **TokenBufferMemory** | Recent messages (token budget) | Bounded by cost | May drop important info |
| **SummaryMemory** | Summary of history | Compact; long context | Summarisation latency; slight distortion |

### Examples

**Buffer (Full History)**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "What's your name?"},
    {"output": "I'm Claude."}
)
memory.save_context(
    {"input": "What's your color?"},
    {"output": "I don't have a color."}
)

print(memory.buffer)
# "Human: What's your name?\nAI: I'm Claude.\nHuman: What's your color?\nAI: I don't have a color."
```

**Window (Last k)**
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=1)  # Only last 1 exchange
# ... after 3 exchanges, the model won't remember the first one
```

**Token Budget**
```python
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=100  # Keep history within 100 tokens
)
```

**Summary**
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
# After each exchange, the LLM summarises the history to save space
```

## Notebooks in This Module

| Notebook | Topics |
|----------|--------|
| `M3_LangChain_Chains.ipynb` | LLMChain, TransformChain, SequentialChain, LCEL, RunnableBranch |
| `M3_LangChain_Agents_intro.ipynb` | Tools, @tool decorator, agents, ReAct loop, built-in tools |
| `M3_LangChain_Memory.ipynb` | ConversationChain, BufferMemory, WindowMemory, TokenBufferMemory, SummaryMemory |
| `Задача_3_1.ipynb` | Exercise notebook for chains (in Russian) |
| `3_2_Решение_задач.ipynb` | Solutions for chains exercises (in Russian) |
| `3_3_Решение_задач.ipynb` | Solutions for agents/memory exercises (in Russian) |

## Key Design Patterns

### LCEL Pipe Syntax
```python
chain = prompt | llm | parser
result = chain.invoke({"var": "value"})
result = chain.stream({"var": "value"})  # Streaming
results = chain.batch([{"var": "a"}, {"var": "b"}])  # Batch
```

### Nested Chains
```python
from langchain.schema.runnable import RunnablePassthrough
from operator import itemgetter

chain = (
    {"context": itemgetter("docs"), "query": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
```

### Tool Documentation Matters
```python
@tool
def search(query: str) -> str:
    """Search Wikipedia for information.

    Args:
        query: The search term, e.g. 'Albert Einstein' or 'Machine Learning'

    Returns:
        Relevant Wikipedia excerpts or error message.
    """
    # implementation
```

The LLM reads this docstring to decide if this tool is useful for a given task.

## Common Pitfalls

- ❌ Agents without `max_iterations` (infinite loops)
- ❌ Tools with unclear docstrings (LLM won't use them)
- ❌ Memory without cleanup (costs grow unbounded)
- ❌ Mixing old `LLMChain` syntax with new LCEL
- ✅ Use LCEL; provide clear tool descriptions; bound memory; add iteration limits
