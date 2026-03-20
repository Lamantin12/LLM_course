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
