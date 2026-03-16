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
llm_factual = ChatOpenAI(temperature=0.0)
response = llm_factual.invoke("What is the capital of France?")
# Output: "Paris" (always the same)

llm_creative = ChatOpenAI(temperature=1.0)
response = llm_creative.invoke("Write a haiku about AI")
# Output: varies each time; may be poetic or odd
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

# Estimate cost: gpt-3.5-turbo is ~$0.0005 per 1K input tokens
cost = len(tokens) / 1000 * 0.0005
print(f"Cost: ${cost:.6f}")
```

## LangChain Prompting

### PromptTemplate (Reusable Prompts)

Instead of f-strings, use `PromptTemplate` for production code:

```python
from langchain import PromptTemplate

template = """You are a {style} writer.
Write about: {topic}"""

prompt = PromptTemplate(
    input_variables=["style", "topic"],
    template=template
)

formatted = prompt.format(style="professional", topic="machine learning")
print(formatted)
```

**Why?** Templates are reusable, testable, and safer than string formatting.

### ChatPromptTemplate (Chat Models)

For chat-based models, use `ChatPromptTemplate` to generate `HumanMessage` and `AIMessage` objects:

```python
from langchain import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\nUser: {user_input}\nAssistant:"
)

messages = prompt.format_messages(user_input="How are you?")
response = llm.invoke(messages)
```

### FewShotPromptTemplate (Teaching by Example)

Teach the model by showing worked examples:

```python
from langchain import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "2 + 2", "output": "4"},
    {"input": "5 * 3", "output": "15"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Solve the math problem:",
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)

prompt_str = few_shot.format(input="10 / 2")
response = llm.invoke(prompt_str)
# Model is more likely to answer "5" because it saw the pattern
```

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
from langchain import PromptTemplate

schemas = [
    ResponseSchema(name="sentiment", description="positive, negative, or neutral"),
    ResponseSchema(name="confidence", description="0–1"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

prompt = PromptTemplate(
    template="Analyze sentiment:\n{format_instructions}\n\nReview: {review}",
    input_variables=["review"],
    partial_variables={"format_instructions": format_instructions}
)

text = "This product is amazing!"
prompt_str = prompt.format(review=text)
response = llm.invoke(prompt_str)

parsed = parser.parse(response.content)
print(parsed)  # {"sentiment": "positive", "confidence": 0.95}
```

The parser **injects instructions** into the prompt telling the model to emit JSON, then **parses the JSON** back into a Python dict.

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
