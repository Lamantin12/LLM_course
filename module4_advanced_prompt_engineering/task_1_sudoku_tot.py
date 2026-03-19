"""
Exercise 4.3.6 — Sudoku 4x4 solver using Tree of Thoughts (ToT)
================================================================
Implements a ToTChecker that validates intermediate and final states
of a 4x4 Sudoku puzzle, then runs ToTChain to find the solution.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
from langchain_experimental.tot.base import ToTChain
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity


class MyChecker(ToTChecker):
    """Checker for 4x4 Sudoku puzzle.

    Grid format: "a,b,c,d|e,f,g,h|i,j,k,l|m,n,o,p"
    where | separates rows and , separates cells.
    * means unfilled cell, digits 1-4 are valid values.
    """

    def evaluate(self, problem_description: str, thoughts: tuple[str, ...] = ()) -> ThoughtValidity:
        last_thought = thoughts[-1]

        # Extract the grid portion (last line that looks like a grid)
        grid_str = last_thought.strip().split("\n")[-1].strip()
        # Also try to find a grid pattern anywhere in the thought
        for line in reversed(last_thought.strip().split("\n")):
            line = line.strip()
            if "|" in line and "," in line:
                grid_str = line
                break

        rows = grid_str.split("|")
        if len(rows) != 4:
            return ThoughtValidity.INVALID

        grid: list[list[str]] = []
        for row in rows:
            cells = [c.strip() for c in row.split(",")]
            if len(cells) != 4:
                return ThoughtValidity.INVALID
            grid.append(cells)

        has_star = any(cell == "*" for row in grid for cell in row)

        # Check all filled cells are valid digits 1-4
        for row in grid:
            for cell in row:
                if cell != "*" and cell not in ("1", "2", "3", "4"):
                    return ThoughtValidity.INVALID

        # Check no duplicates in rows (among filled cells)
        for row in grid:
            filled = [c for c in row if c != "*"]
            if len(filled) != len(set(filled)):
                return ThoughtValidity.INVALID

        # Check no duplicates in columns
        for col_idx in range(4):
            filled = [grid[row_idx][col_idx] for row_idx in range(4) if grid[row_idx][col_idx] != "*"]
            if len(filled) != len(set(filled)):
                return ThoughtValidity.INVALID

        # Check no duplicates in 2x2 subgrids
        for box_row in range(2):
            for box_col in range(2):
                filled = []
                for r in range(2):
                    for c in range(2):
                        cell = grid[box_row * 2 + r][box_col * 2 + c]
                        if cell != "*":
                            filled.append(cell)
                if len(filled) != len(set(filled)):
                    return ThoughtValidity.INVALID

        if has_star:
            return ThoughtValidity.VALID_INTERMEDIATE
        return ThoughtValidity.VALID_FINAL


# ── Assertions from the notebook ──

def run_tests():
    checker = MyChecker()
    assert checker.evaluate("", ("3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1",)) == ThoughtValidity.VALID_INTERMEDIATE
    assert checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1",)) == ThoughtValidity.VALID_FINAL
    assert checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,*,1",)) == ThoughtValidity.VALID_INTERMEDIATE
    assert checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,*,3,1",)) == ThoughtValidity.INVALID
    print("All assertions passed!")


# ── Main ──

if __name__ == "__main__":
    run_tests()

    llm = ChatOpenAI(
        api_key=API_KEY,
        model="gpt-4o-mini",
        base_url="https://api.vsellm.ru/",
    )

    sudoku_puzzle = "3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1"
    problem_description = f"""{sudoku_puzzle}
- Это головоломка Судоку размером 4x4.
- Символ * обозначает ячейку, которую тебе нужно заполнить.
- Символ | разделяет строки.
- На каждом шаге замени одну или несколько * цифрами от 1 до 4.
- Напиши текущее состояние пазла с добавленными цифрами
- Ни в одной строке, столбце или 2x2 подсетке не должно быть одинаковых цифр.
- Сохраняй полученные цифры из предыдущих успешных мыслей на своих местах.
- Каждая мысль может быть промежуточным или окончательным решением.""".strip()

    tot_chain = ToTChain(
        llm=llm,
        checker=MyChecker(),
        k=30,
        c=2,
        verbose=True,
        verbose_llm=False,
    )
    result = tot_chain.invoke({"problem_description": problem_description})
    print(result)
