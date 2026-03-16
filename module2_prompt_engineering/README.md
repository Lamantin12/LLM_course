# Module 2: Prompt Engineering

Master the craft of writing effective prompts. Learn how to structure instructions, inject context, control model behaviour with temperature, extract structured data, and use few-shot examples to guide the model.

## What You'll Learn

- The four-part prompt structure (Instructions, Context, Input, Output Indicator)
- How `temperature` affects output randomness and creativity
- Token counting and API cost estimation
- `PromptTemplate` for reusable, parameterised prompts
- `ChatPromptTemplate` for chat-based conversations
- `FewShotPromptTemplate` for teaching the model by example
- `StructuredOutputParser` for converting LLM text into Python dicts
- Dynamic few-shot example selection

## The Four-Part Prompt Structure

A well-structured prompt has four components:

```
[INSTRUCTIONS]
You are a sentiment analyzer. Classify the review sentiment as "positive", "negative", or "neutral".

[CONTEXT]
You have access to these product reviews:

[INPUT]
"This product broke after one week. Total waste of money."

[OUTPUT INDICATOR]
Sentiment:
```

### Component Details

| Part | Purpose | Example |
|------|---------|---------|
| **Instructions** | Define the task and role | "You are a helpful assistant" |
| **Context** | Provide background info or few-shot examples | Retrieved documents or worked examples |
| **Input** | The user's question or data | "Classify this review..." |
| **Output Indicator** | Tell the model what form output should take | "Answer in JSON: {\"sentiment\": ...}" |

## Temperature Effects

`temperature` controls how "creative" vs. "deterministic" the model is.

```python
from utils import ChatOpenAI

llm_factual = ChatOpenAI(temperature=0.0, course_api_key=course_api_key)
response = llm_factual.invoke("Привет, как дела?")
# Output: always the same deterministic response

llm_creative = ChatOpenAI(temperature=1.0, course_api_key=course_api_key)
response = llm_creative.invoke("Привет, как дела?")
# Output: varies each time; more creative and diverse
```

### When to Use Which

| Temperature | Use Case | Example |
|------------|----------|---------|
| **0.0–0.3** | Factual, deterministic | QA, translation, classification |
| **0.5–0.7** | Balanced | Chatbots, writing assistance |
| **0.8–1.5** | Creative, diverse | Poetry, brainstorming, code generation |

## Token Counting & Cost Estimation

```python
import tiktoken

encoder = tiktoken.get_encoding("p50k_base")
tokens = encoder.encode("What is the capital of France?")
print(len(tokens))  # ~8 tokens
```

**Cost estimation** (from the notebook): a prompt + response of ~1000 tokens costs roughly `$0.002`. So 10 users making 100 requests each would burn through `$2`, while only using ~1/3 of the context window. Plan your service's economics accordingly.

## LangChain Prompting

### PromptTemplate (Reusable Prompts)

Instead of f-strings, use `PromptTemplate` for production code:

```python
from langchain import PromptTemplate

template = """Ответь на вопрос, опираясь на контекст ниже.
Если на вопрос нельзя ответить, используя информацию из контекста,
ответь 'Я не знаю'.

Context: {context}

Question: {query}

Answer: """

prompt = PromptTemplate(input_variables=["context", "query"], template=template)

formatted = prompt.format(
    context="В России лидером онлайн курсов является Stepik.",
    query="Какая платформа популярна в России?"
)
print(llm.invoke(formatted).content)  # → "Stepik"
```

**Why?** Templates are reusable, testable, and safer than string formatting.

### ChatPromptTemplate (Chat Models)

For chat-based models, use `ChatPromptTemplate` to generate `HumanMessage` and `AIMessage` objects:

```python
from langchain.prompts import ChatPromptTemplate

template = """Ответь на вопрос, опираясь на контекст ниже.

Context: {context}

Question: {query}

Answer: """

prompt = ChatPromptTemplate.from_template(template)
messages = prompt.format_messages(context="Ламы водятся в Перу.", query="Где водятся ламы?")
# Returns a list of HumanMessage objects
print(llm.invoke(messages).content)  # → "В Перу."
```

Key difference from `PromptTemplate`: `ChatPromptTemplate` returns `HumanMessage`/`AIMessage` objects (chat-mode), while `PromptTemplate` returns raw strings.

### FewShotPromptTemplate (Teaching by Example)

Teach the model by showing worked examples:

```python
from langchain import FewShotPromptTemplate, PromptTemplate

# Sarcastic chatbot example from the notebook
examples = [
    {"query": "Как дела?", "answer": "Не могу пожаловаться, но иногда всё-таки жалуюсь."},
    {"query": "Сколько время?", "answer": "Самое время купить часы."},
]

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template="User: {query}\nAI: {answer}\n"
)

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Это разговор с ИИ-помощником. Помощник саркастичен и остроумен.\nВот несколько примеров:",
    suffix="\nUser: {query}\nAI: ",
    input_variables=["query"],
    example_separator="\n\n"
)

print(llm.invoke(few_shot.format(query="Почему падает снег?")).content)
# → "Потому что небо не умеет держать себя в руках."
```

**Why few-shot works**: Without examples, asking `A + A = ?` gives `2A` (mathematical). With examples like `A + A = AA`, `B + С = BC`, the model switches to concatenation and answers `1 + 1 = 11`. The examples steer the model's interpretation.

### LengthBasedExampleSelector (Dynamic Few-Shot)

Auto-trim few-shot examples based on query length:

```python
from langchain.prompts import LengthBasedExampleSelector, FewShotPromptTemplate, PromptTemplate

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50  # Token budget
)

few_shot = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix="Solve:",
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)
```

If the query is long, fewer examples are included to stay within the 50-token budget.

## Structured Output Parsing

Extract structured data (dicts, JSON) from LLM responses:

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate

# Define the fields we want extracted
schemas = [
    ResponseSchema(name="gift", description="Был ли товар куплен в подарок? True/False"),
    ResponseSchema(name="delivery_days", description="Сколько дней доставка? -1 если неизвестно"),
    ResponseSchema(name="price_value", description="Оценка стоимости товара"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

template = """Из следующего текста извлеки информацию:
text: {text}
{format_instructions}"""

prompt = ChatPromptTemplate.from_template(template)
messages = prompt.format_messages(text=customer_review, format_instructions=format_instructions)
response = llm.invoke(messages)

# LLM returns a string, but the parser converts it to a dict
output_dict = parser.parse(response.content)
print(type(output_dict))      # <class 'dict'>
print(output_dict.get("gift")) # "True"
```

**How it works**: `get_format_instructions()` generates a markdown JSON schema that's injected into the prompt, telling the model to emit a specific JSON format. The parser then extracts the JSON and returns a Python `dict`.

## Notebooks in This Module

| Notebook | Topics |
|----------|--------|
| `M2_1_Prompt_Engineering_intro.ipynb` | Prompt structure, temperature, token counting, parametric vs. source knowledge |
| `M2_2_LangChain_Prompting.ipynb` | PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate, StructuredOutputParser |
| `2_1_task_solution.ipynb` | Solutions to module 2.1 exercises |
| `2_2_Решение_задач.ipynb` | Solutions to module 2.2 exercises (in Russian) |

## Key Principles

1. **Clarity over brevity**: Be explicit. The model responds better to clear instructions.
2. **Output indicators matter**: Telling the model "Answer in JSON" or "Respond in one sentence" shapes the output.
3. **Examples teach**: Few-shot prompting is often more effective than detailed instructions.
4. **Temperature is your control**: Use low temperature for facts, high for creativity.
5. **Parametric knowledge isn't always right**: Always include source knowledge if accuracy matters.

## Common Pitfalls

- ❌ Using f-strings instead of `PromptTemplate` in production
- ❌ Forgetting to include output format instructions
- ❌ Using high temperature for factual tasks
- ❌ Injecting too much context (dilutes retrieval quality)
- ✅ Structure prompts clearly; show examples; control temperature; parse outputs consistently
