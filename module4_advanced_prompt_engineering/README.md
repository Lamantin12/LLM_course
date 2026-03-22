# Module 4 — Advanced Prompt Engineering

Go beyond basic prompting: self-consistency voting, knowledge-augmented generation, tree-search reasoning, program-aided calculation, and emotionally-tuned prompts.

---

## Topics Covered

- **Self-Consistency** — generate the same question multiple times and pick the most frequent answer via majority vote
- **Generated Knowledge Prompting** — multi-step pipeline where the model first generates context, then produces a richer final answer
- **Tree of Thoughts (ToT)** — explore reasoning as a tree with BFS/DFS and backtracking; requires a custom `ToTChecker`
- **PAL (Program-Aided Language Models)** — LLM writes Python code that is executed by an interpreter instead of doing arithmetic in text
- **Emotional Prompting** — adding emotional stakes to prompts measurably improves output quality

---

## Lecture Notes

### M4_Advanced_Prompting.ipynb

**Self-Consistency** — Runs the same prompt multiple times at high temperature, then takes the majority answer via voting.
`Counter(answers).most_common(1)[0][0]` aggregates N independent model calls into a single voted answer, reducing variance from a single stochastic sample. The critical limitation shown in the notebook: if the model is consistently wrong (all 5 runs return the same wrong answer), majority voting cannot fix it — the technique only helps when the model is right most of the time.
```python
from collections import Counter
answers = [chain.invoke({"question": q}) for _ in range(5)]
majority = Counter(answers).most_common(1)[0][0]
```

**Generated Knowledge** — Multi-step pipeline where the model first generates context, then uses that context for a richer final answer.
`chain1` generates N topic-related questions; `chain2.batch()` answers them in parallel using domain-expert personas; `chain3` synthesises the answers into a final artifact. The result is noticeably richer than a single-shot prompt because the context window now contains model-generated knowledge rather than just the user's brief query.
```python
questions = questions_chain.invoke({"topic": "black holes"})
answers = expert_chain.batch([{"q": q} for q in questions])
article = synthesis_chain.invoke({"knowledge": "\n".join(answers), "topic": "black holes"})
```

**Tree of Thoughts** — Explores reasoning as a tree with branching and backtracking, validated at each intermediate step.
`MyChecker(ToTChecker)` implements `evaluate()` returning `VALID_INTERMEDIATE`, `VALID_FINAL`, or `INVALID`; `ToTChain(checker=checker, k=50, c=3)` runs BFS. The non-obvious implementation requirement: the checker parses the model's plain-text output with regex — the `VALID_INTERMEDIATE` pattern must match exactly what the LLM produces.
```python
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.base import ToTChain

class MyChecker(ToTChecker):
    def evaluate(self, problem_description, current_state, next_move):
        # parse model output; return VALID_FINAL / VALID_INTERMEDIATE / INVALID
        ...

chain = ToTChain(llm=llm, checker=MyChecker(), k=50, c=3, verbose=True)
result = chain.run(problem_description="Game of 24: 2 3 4 5")
```

**PAL** — Program-Aided Language models — the LLM generates Python code that an interpreter executes instead of doing arithmetic in text.
`PALChain.from_math_prompt(llm, allow_dangerous_code=True)` prompts the model to write a `solution()` function, then executes it and returns the numeric result. This eliminates token-level arithmetic errors (e.g. vegetable-counting mistakes); `allow_dangerous_code=True` is required — only use in sandboxed/trusted environments.
```python
from langchain_experimental.pal_chain import PALChain
pal_chain = PALChain.from_math_prompt(llm, allow_dangerous_code=True, verbose=True)
result = pal_chain.invoke("If there are 3 carrots and 5 potatoes, how many vegetables?")
print(result["result"])   # "8"
```

**Emotional Prompting** — Adding emotional stakes to a prompt measurably improves response quality and length.
Appending phrases like `"This is very important to my career"` or a tip offer raises the model's implicit priority for the request, producing longer and more thorough responses. The notebook benchmarks multiple phrasings and finds diminishing returns above ~$20 in tip offers — real effect, but not a substitute for good prompt structure.
```python
base_prompt = "Summarise this document: {text}"
emotional_prompt = "Summarise this document: {text}\n\nThis is critical for my PhD thesis defence."
# Emotional version produces ~30% longer, more detailed summaries
```

---

## Files

| File | Description |
|------|-------------|
| `M4_Advanced_Prompting.ipynb` | Lecture notebook — theory, examples, and output for all 5 techniques |
| `M4_3_exercises.ipynb` | Exercise notebook — tasks 4.3.6 (Sudoku ToT) and 4.3.7 (PAL math) |
| `advanced_prompting.py` | All lecture examples as a runnable script |
| `task_1_sudoku_tot.py` | **Exercise 4.3.6** — 4×4 Sudoku solver with `MyChecker` and `ToTChain` |
| `task_2_pal_math.py` | **Exercise 4.3.7** — PAL math solver over a CSV dataset |

---

## How to Run

```bash
pip install langchain langchain_experimental langchain-openai openai pandas tqdm python-dotenv

# Lecture demos
python module4_advanced_prompt_engineering/advanced_prompting.py

# Exercise 4.3.6 — Sudoku ToT
python module4_advanced_prompt_engineering/task_1_sudoku_tot.py

# Exercise 4.3.7 — PAL math (saves results to submissions/m4adv_3_7_solution.csv)
python module4_advanced_prompt_engineering/task_2_pal_math.py
```

API key is loaded from `.env` in the repo root: `OPENAI_API_KEY=your_key_here`

---

## Key Concepts

- **Self-Consistency**: Run the same prompt N times with temperature > 0, then take the majority answer — reduces variance from a single stochastic sample.
- **Generated Knowledge**: Prime the model with self-generated context before the final task — acts like "warming up" the relevant knowledge in the context window.
- **Tree of Thoughts (ToT)**: Each node is an intermediate reasoning step; the algorithm explores branches and backtracks from dead ends — effective for puzzles with verifiable intermediate states.
- **PAL**: The LLM generates Python code instead of prose answers; the interpreter handles exact arithmetic — eliminates token-level calculation errors.
- **`ToTChecker`**: A custom class that validates each reasoning step in a ToT chain; returns `True/False` to guide the search.
- **`PALChain.from_math_prompt`**: LangChain wrapper that prompts the model for code and executes it; requires `allow_dangerous_code=True` in a controlled environment.
- **Emotional Prompting**: Stakes like "my job depends on it" or a tip offer measurably increase response length and quality; diminishing returns above ~$20.

---

## References

- [Self-Consistency — Wang et al. (2022)](https://arxiv.org/abs/2203.11171)
- [Generated Knowledge — Liu et al. (2022)](https://arxiv.org/abs/2110.08387)
- [Tree of Thoughts — Yao et al. (2023)](https://arxiv.org/abs/2305.10601)
- [PAL — Gao et al. (2022)](https://arxiv.org/abs/2211.10435)
- [Emotional Prompting — Li et al. (2023)](https://arxiv.org/abs/2307.11760)
