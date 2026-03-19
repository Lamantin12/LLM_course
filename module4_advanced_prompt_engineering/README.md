# Module 4 — Advanced Prompt Engineering

## Topics Covered

### Self-Consistency
Generate the same question multiple times and pick the most frequent answer via majority vote (`Counter`). Reduces random hallucinations — effective when the model gets the right answer *at least sometimes*. Won't help if the model is consistently wrong.

**When to use:** Kaggle LLM competitions, factual Q&A, any task where variance is the problem rather than systematic bias.

### Generated Knowledge Prompting
Build a multi-step pipeline where the model first generates context about a topic, then uses that context to produce a richer final answer. Pipeline in the lecture:
1. Generate interesting questions about a topic
2. Answer each question as an expert (using `chain.batch`)
3. Combine Q&A pairs into a blog article

**When to use:** Content generation, research summaries, any task that benefits from "warming up" the context before the final output.

### Tree of Thoughts (ToT)
Explores reasoning as a tree: each node is an intermediate thought (text step). Combines LLM generation with search algorithms (BFS/DFS) and allows backtracking from dead ends. Requires a custom `ToTChecker` to validate each step.

Key parameters in `ToTChain`:
- `k` — max number of iterations
- `c` — branching factor (thoughts per node)

**When to use:** Multi-step puzzles (Game of 24, Sudoku, crosswords), any task with verifiable intermediate steps.

### PAL (Program-Aided Language Models)
Instead of reasoning in text, the LLM writes Python code which is then executed by an interpreter. Avoids arithmetic errors and complex date/unit calculations. Uses `PALChain.from_math_prompt`.

⚠️ **Security note:** PAL executes arbitrary Python — always run in an isolated environment. Use `allow_dangerous_code=True` only in controlled settings.

**When to use:** Math word problems, date arithmetic, counting tasks, any calculation-heavy problem.

### Emotional Prompting
Adding emotional context to prompts ("my job depends on it", "tip $10-20 for a correct answer") measurably improves output quality and length. Based on [Li et al. (2023)](https://arxiv.org/abs/2307.11760).

Note: tips above ~$20 show no further improvement.

---

## Files

| File | Description |
|------|-------------|
| `M4_Basic_Advansic_Prompting.ipynb` | Lecture notebook — theory, examples, and output for all 5 techniques |
| `задачи_4_3_.ipynb` | Exercise notebook — tasks 4.3.6 (Sudoku ToT) and 4.3.7 (PAL math) |
| `advanced_prompting.py` | All lecture examples as a runnable script |
| `task_1_sudoku_tot.py` | **Exercise 4.3.6** — 4x4 Sudoku solver with `MyChecker` and `ToTChain` |
| `task_2_pal_math.py` | **Exercise 4.3.7** — PAL math solver over a CSV dataset, outputs `5.6.7_solution.csv` |

---

## How to Run

```bash
pip install langchain langchain_experimental langchain-openai openai pandas tqdm python-dotenv

# Lecture demos
python advanced_prompting.py

# Exercise 4.3.6 — Sudoku (runs assertions first, then ToT)
python task_1_sudoku_tot.py

# Exercise 4.3.7 — PAL math (downloads CSV, saves results to 5.6.7_solution.csv)
python task_2_pal_math.py
```

API key is loaded from `.env` in the repo root: `OPENAI_API_KEY=your_key_here`

---

## Key References

- [Self-Consistency — Wang et al. (2022)](https://arxiv.org/abs/2203.11171)
- [Generated Knowledge — Liu et al. (2022)](https://arxiv.org/abs/2110.08387)
- [Tree of Thoughts — Yao et al. (2023)](https://arxiv.org/abs/2305.10601)
- [PAL — Gao et al. (2022)](https://arxiv.org/abs/2211.10435)
- [Emotional Prompting — Li et al. (2023)](https://arxiv.org/abs/2307.11760)
- [Prompting techniques overview](https://www.promptingguide.ai/techniques)