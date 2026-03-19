# Module 4 — Advanced Prompt Engineering

## Topics Covered

| Technique | Description |
|-----------|-------------|
| **Self-Consistency** | Generate multiple answers, pick the most frequent — fights random hallucinations |
| **Generated Knowledge Prompting** | Multi-step pipeline: generate context first, then use it to produce a richer final answer |
| **Tree of Thoughts (ToT)** | Explore a tree of intermediate reasoning steps with backtracking (BFS/DFS) |
| **PAL (Program-Aided Language)** | Let the LLM write Python code instead of reasoning in text — avoids arithmetic errors |
| **Emotional Prompting** | Adding emotional pressure / tips to improve generation quality |

## Files

| File | Description |
|------|-------------|
| `M4_Basic_Advansic_Prompting.ipynb` | Lecture notebook with theory and examples |
| `задачи_4_3_.ipynb` | Exercise notebook (tasks 4.3.6 & 4.3.7) |
| `advanced_prompting.py` | All lecture examples extracted into a runnable script |
| `sudoku_tot.py` | **Exercise 4.3.6** — 4x4 Sudoku solver using ToT with a custom checker |
| `pal_math.py` | **Exercise 4.3.7** — Solving math word problems with PALChain |

## How to Run

```bash
pip install langchain langchain_experimental langchain-openai openai pandas tqdm

# Lecture demos
python advanced_prompting.py

# Exercise 4.3.6 — Sudoku (runs assertions first, then ToT)
python sudoku_tot.py

# Exercise 4.3.7 — PAL math (downloads CSV, saves results to 5.6.7_solution.csv)
python pal_math.py
```

All scripts prompt for the course API key at startup.

## Key References

- [Self-Consistency — Wang et al. (2022)](https://arxiv.org/abs/2203.11171)
- [Generated Knowledge — Liu et al. (2022)](https://arxiv.org/abs/2110.08387)
- [Tree of Thoughts — Yao et al. (2023)](https://arxiv.org/abs/2305.10601)
- [PAL — Gao et al. (2022)](https://arxiv.org/abs/2211.10435)
- [Emotional Prompting — Li et al. (2023)](https://arxiv.org/abs/2307.11760)
