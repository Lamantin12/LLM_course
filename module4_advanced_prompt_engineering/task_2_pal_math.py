"""
Exercise 4.3.7 — PAL vs Math Problems
======================================
Uses PALChain to solve math word problems from a CSV dataset,
letting the LLM generate Python code instead of reasoning in text.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_experimental.pal_chain.base import PALChain


def main():
    llm = ChatOpenAI(
        api_key=API_KEY,
        model="gpt-4o-mini",
        base_url="https://api.vsellm.ru/",
    )

    df = pd.read_csv("https://stepik.org/media/attachments/lesson/1282589/math_pal.csv")

    pal_chain = PALChain.from_math_prompt(llm, allow_dangerous_code=True, verbose=True)

    add = (
        "\n\nIMPORTANT: Generate ONLY the Python code. DO NOT include ```python or ``` "
        "DO NOT use `import` statements."
        "Use pure Python only."
        "If you need π use: 3.141592653589793"
        "If you need sqrt use: **0.5"
        "markdown fences or any other extra text. Just the raw Python function `solution()`."
        "THe final answer that you will get should be just a number float or int"
    )

    answers = []
    for text_input in tqdm(df["task"]):
        prompt = f"\n{text_input}{add}"
        try:
            result = pal_chain.invoke(prompt)
            answer = float(result["result"])
        except Exception as e:
            print(f"Error on task: {text_input[:60]}... -> {e}")
            answer = 0.0
        answers.append(answer)

    df["answer"] = answers
    df[["task", "answer"]].to_csv("m4adv_3_7_solution.csv", index=False)
    print("Results saved to m4adv_3_7_solution.csv")


if __name__ == "__main__":
    main()
