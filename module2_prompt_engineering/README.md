# Module 2 — Prompt Engineering

Master the craft of writing effective prompts: structure instructions, inject context, control model behaviour with temperature, extract structured data, and use few-shot examples to guide the model.

---

## Topics Covered

- Four-part prompt structure: Instructions, Context, Input, Output Indicator
- `temperature` — controlling output randomness and creativity
- Token counting and API cost estimation with `tiktoken`
- `PromptTemplate` — reusable, parameterised prompts
- `ChatPromptTemplate` — chat-based conversations
- `FewShotPromptTemplate` — teaching the model by example
- `LengthBasedExampleSelector` — dynamic few-shot trimming
- `StructuredOutputParser` — converting LLM text into Python dicts

---

## Lecture Notes

### M2_1_Prompt_Engineering_intro.ipynb

**Four-part prompt structure** — A systematic template that separates role/task, background, user input, and expected output into distinct blocks.
Context paragraphs are separated from the prompt body with `###` delimiters to prevent the model from mixing up what is background versus what is the question; ending with `"Answer:"` shapes the output format without extra parsing code. Adding a fallback instruction (`"If you don't know, say 'I don't know'"`) to the instructions block measurably reduces hallucination.
```python
prompt = """You are a helpful assistant. Answer based on the context below.

###Context: {context}

###Question: {question}

Answer:"""
```

**Temperature** — Scalar that controls how randomly the model samples from its token probability distribution.
`temperature=0.0` makes the model deterministic (always picks the highest-probability token); `temperature=1.0+` produces creative, varied, and sometimes incoherent output. The demo compares a `0.0` vs `1.0` sarcastic chatbot — the higher-temperature response is more varied but less reliable; use 0.0–0.3 for tasks where correctness matters.
```python
client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    temperature=0.0   # deterministic; use 1.0 for creative tasks
)
```

**tiktoken** — OpenAI's tokenizer library for counting tokens before making an API call.
`tiktoken.get_encoding("p50k_base").encode(text)` returns a token list; `len(...)` gives the exact count for cost estimation client-side before the API call. The notebook shows that a realistic Q&A exchange consumes ~1000 tokens (~$0.002), motivating careful prompt economy.
```python
import tiktoken
enc = tiktoken.get_encoding("p50k_base")
tokens = enc.encode("Hello, world!")
print(len(tokens))   # 4 tokens
```

### M2_2_LangChain_Prompting.ipynb

**PromptTemplate** — LangChain's reusable prompt with named `{variable}` placeholders, replacing fragile f-strings.
`PromptTemplate(input_variables=["topic"], template="Write about {topic}")` separates the template from the data; `.format(topic="cats")` renders it. Unlike f-strings, templates are testable, serializable to disk, and safe to compose — `FewShotPromptTemplate` and `ChatPromptTemplate` both build on this base class.
```python
from langchain.prompts import PromptTemplate
template = PromptTemplate(
    input_variables=["style", "topic"],
    template="Write a {style} poem about {topic}."
)
print(template.format(style="haiku", topic="autumn"))
```

**FewShotPromptTemplate** — Extends PromptTemplate to inject worked examples between a prefix and the user query.
A list of `{query, answer}` dicts becomes the examples block; `example_prompt` defines how each pair is formatted; `suffix` holds the live query slot. The model reads the examples as implicit rules — more robust than describing the rules in prose. The gotcha: examples are hardcoded; use an `ExampleSelector` when the example set is large.
```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
example_prompt = PromptTemplate(
    input_variables=["query", "answer"], template="Q: {query}\nA: {answer}")
few_shot = FewShotPromptTemplate(
    examples=[{"query": "happy", "answer": "sad"}],
    example_prompt=example_prompt,
    prefix="Give the antonym of each word.",
    suffix="Q: {input}\nA:", input_variables=["input"])
```

**LengthBasedExampleSelector** — Dynamically trims the example list to fit within a word budget.
`LengthBasedExampleSelector(examples=examples, example_prompt=ep, max_length=50)` measures each example's word count and includes as many as fit; longer queries leave less room for examples. The gotcha: `max_length` counts words, not tokens — may still exceed API limits for multilingual text.
```python
from langchain.prompts.example_selector import LengthBasedExampleSelector
selector = LengthBasedExampleSelector(
    examples=examples, example_prompt=example_prompt, max_length=50
)
few_shot = FewShotPromptTemplate(
    example_selector=selector, example_prompt=example_prompt,
    prefix="...", suffix="Q: {input}\nA:", input_variables=["input"])
```

**StructuredOutputParser** — Injects a JSON schema into the prompt and parses the LLM's text response into a Python dict.
`ResponseSchema` defines each field name and description; `get_format_instructions()` generates the schema string to embed in the prompt; `.parse(response.content)` returns a typed dict. The gotcha: the model can still produce malformed JSON — wrap `.parse()` in a try/except or use `PydanticOutputParser` for stricter validation.
```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
schemas = [ResponseSchema(name="gift", description="Was it a gift? True/False"),
           ResponseSchema(name="price", description="Price, or 'unknown'")]
parser = StructuredOutputParser.from_response_schemas(schemas)
prompt = f"Extract info.\n{parser.get_format_instructions()}\nText: {{text}}"
result = parser.parse(llm.invoke(prompt.format(text="...")))
print(result["gift"])
```

---

## Files

| File | Description |
|------|-------------|
| `M2_1_Prompt_Engineering_intro.ipynb` | Prompt structure, temperature, token counting, parametric vs. source knowledge |
| `M2_2_LangChain_Prompting.ipynb` | PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate, StructuredOutputParser |
| `M2_1_exercises.ipynb` | Exercises for section 2.1 |
| `M2_2_exercises.ipynb` | Exercises for section 2.2 |

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook module2_prompt_engineering/M2_1_Prompt_Engineering_intro.ipynb
```

---

## Key Concepts

- **Four-Part Prompt**: Instructions (task/role) + Context (background) + Input (user data) + Output Indicator (expected format).
- **Temperature**: 0.0–0.3 for factual/deterministic tasks; 0.5–0.7 for balanced; 0.8–1.5 for creative output.
- **Few-Shot Prompting**: Showing worked examples steers the model's interpretation more effectively than detailed instructions alone.
- **PromptTemplate**: Reusable, testable prompt with `{variable}` slots — prefer over f-strings in production.
- **StructuredOutputParser**: Injects a JSON schema into the prompt and parses the LLM's text response into a Python `dict`.
- **`format_instructions`**: Auto-generated by the parser; tells the model the exact JSON format to emit.
- **Output Indicator**: Ending the prompt with "Sentiment:" or "Answer in JSON:" shapes output format without explicit parsing code.
