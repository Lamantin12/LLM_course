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
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils import ChatOpenAI

llm = ChatOpenAI(temperature=0.0, course_api_key=course_api_key)

template = '''Перепиши текcт в заданном стиле.
Текст:{output_text}
Стиль: {style}.
Результат:'''

prompt = PromptTemplate(input_variables=['output_text', 'style'], template=template)
chain = LLMChain(llm=llm, prompt=prompt, output_key='final_output')

result = chain.invoke({'output_text': text, 'style': 'Rap'})
print(result['final_output'])
```

### LCEL (Modern Syntax)

LCEL is cleaner, more composable, and handles streaming natively:

```python
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from utils import ChatOpenAI

llm = ChatOpenAI(temperature=0.0, course_api_key=course_api_key)

template = '''Перепиши этот текcт в заданном стиле: {output_text}
Стиль: {style}.
Результат:'''

prompt = PromptTemplate(input_variables=['output_text', 'style'], template=template)

chain = prompt | llm  # That's it!

# With parser for string output:
chain_with_parser = prompt | llm | StrOutputParser()
```

**Benefit**: One-line chain definition vs. 10+ lines of setup code. The notebook literally says "И ВСЁ!" (And that's it!)

### TransformChain (Pure Python Functions)

Insert Python logic into a chain:

```python
import re
from langchain.chains import TransformChain

def del_spaces(inputs: dict) -> dict:
    text = inputs["text"]
    text = re.sub(r'(\r\n|\r|\n){2,}', r'\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return {"output_text": text}

text_clean_chain = TransformChain(
    input_variables=["text"],
    output_variables=["output_text"],
    transform=del_spaces
)

result = text_clean_chain.invoke(dirty_text)
print(result['output_text'])  # Cleaned text with no extra spaces/newlines
```

Note: `TransformChain` has no LLM — it's pure Python. This saves tokens by cleaning input before sending to the model.

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
from langchain.agents import load_tools, Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool

# Tools supported out-of-the-box by load_tools
tools = load_tools(["arxiv", "wikipedia"], llm=llm)

# DuckDuckGo search
search = DuckDuckGoSearchRun()
tools.append(Tool(name="Search", func=search.run,
                  description="useful for current events"))

# Code execution (writes and runs Python)
python_repl = PythonREPLTool()

# Math via LLMMathChain (DEPRECATED — use agents instead)
from langchain.chains import LLMMathChain
math_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool(name='Calculator', func=math_chain.run,
                 description='Может производить математические расчёты.')
```

> **Utility Chains (e.g. `LLMMathChain`) are deprecated** — the notebook explicitly says "БОЛЬШЕ НЕ ПОДДЕРЖИВАЮТСЯ ФРЭЙМВОРКОМ — их заменили агенты". Use agents with tools instead.

### Initializing an Agent

```python
from langchain.agents import initialize_agent

agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True
)

result = agent("Сколько будет (4.5*2.1)+2.2?")
print(result['output'])  # 11.65 (uses the calculator tool)
```

**Parameters**:
- `verbose=True`: Print reasoning steps (see the ReAct thought/action/observation loop)
- `max_iterations`: Prevent infinite loops
- `handle_parsing_errors=True`: Gracefully handle LLM parsing failures

> Note: `initialize_agent` is marked as deprecated in recent LangChain — use `create_react_agent` for new code. The notebook still uses `initialize_agent`.

### When LLMs Fail

LLMs struggle with tasks that require exact computation or current information:

```python
# ❌ LLM gets this wrong due to tokenisation
llm.invoke("Сколько букв в слове зачёт?")
# → "6 букв" (wrong; it's 5) because of how tokens ≠ letters

# ✅ Use an agent with a custom tool
@tool
def get_word_length(word: str) -> int:
    """Возвращает длину слова"""
    return len(word)

agent = initialize_agent(tools=[get_word_length], llm=llm, verbose=True)
agent("Сколько букв в слове зачёт?")
# → 5 (correct; used the tool)
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

conversation.invoke("Привет, ChatGPT! Меня зовут Иван. Как дела?")
conversation.invoke("Сможешь помочь мне в написании кода на Python?")
conversation.invoke("Как вывести на экран 'Hello, world!' ?")
# → Answers with Python syntax because it *remembers* the earlier request about Python
```

Under the hood, a `ConversationChain` uses a prompt with `{history}` and `{input}` variables. The `memory` automatically populates `{history}` with previous messages.

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
from langchain.chains.conversation.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
# After each exchange, the LLM summarises the history to save space
# Note: summaries are generated in English regardless of conversation language
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
