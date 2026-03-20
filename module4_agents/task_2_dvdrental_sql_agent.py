"""
Task 2: SQL Agent on PostgreSQL DVD Rental Database

Sets up a PostgreSQL database with the dvdrental dataset, runs a SQL agent against it,
answers questions from the course CSV, and saves results.

setup_db() is OS-aware: uses Homebrew on macOS, apt-get/sudo on Linux/Colab.
"""
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

os.environ['KMP_DUPLICATE_LIB_OK']='True'
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = "https://api.vsellm.ru/"
API_KEY = os.getenv("OPENAI_API_KEY")

DB_URI = "postgresql+psycopg2://root:mypass@localhost:5432/dvdrental"
QUESTIONS_URL = "https://stepik.org/media/attachments/lesson/1107866/rental_dvd.csv"
SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=API_KEY, model="gpt-4o-mini", base_url=BASE_URL, temperature=temperature
    )


def build_sql_agent():
    """Create a SQL agent connected to the dvdrental PostgreSQL database."""
    llm = get_llm()
    db = SQLDatabase.from_uri(DB_URI)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    suffix = """I should now give the final answer. 
    IMPORTANT: The final answer must be ONLY the value itself (e.g., '42' or 'email@test.com'). 
    Do not add any conversational text, prefixes, or suffixes. 
    If no result found, say 'Не знаю'."""
    return create_sql_agent(llm, agent_type="tool-calling", toolkit=toolkit, verbose=True, suffix=suffix)


def answer_questions(agent_executor) -> pd.DataFrame:
    """Download questions CSV, run agent on each, return DataFrame with answers."""
    df = pd.read_csv(QUESTIONS_URL)
    questions = df.iloc[:, 0].tolist()

    answers = []
    for q in questions:
        print(f"\nQ: {q}")
        try:
            result = agent_executor.invoke({"input": q})
            answer = result["output"]
        except Exception as e:
            answer = f"ERROR: {e}"
        print(f"A: {answer}")
        answers.append(answer)

    df["answer"] = answers
    return df[["question", "answer"]] if "question" in df.columns else pd.DataFrame(
        {"question": questions, "answer": answers}
    )

def main():
    agent_executor = build_sql_agent()
    results = answer_questions(agent_executor)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSIONS_DIR / "m4agents_step9_solution.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} answers to {output_path}")



if __name__ == "__main__":
    main()
