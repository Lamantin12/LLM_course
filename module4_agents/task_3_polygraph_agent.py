"""
Task 3: Polygraph Agent — True/False fact-checker

Loads statements from a course CSV, uses a Tool Calling Agent to evaluate each one,
and saves results with Python bool answers.

Tools:
- llm-math: for math expressions
- serpapi: for web-based fact verification (if SERPAPI_API_KEY is set)
- HumanInputRun: fallback for questions the agent can't answer automatically
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import load_tools, AgentExecutor, create_tool_calling_agent
from langchain_community.tools.human.tool import HumanInputRun
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = "https://api.vsellm.ru/"
API_KEY = os.getenv("OPENAI_API_KEY")

QUESTIONS_URL = "https://stepik.org/media/attachments/lesson/1110884/questions.csv"
SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"

SYSTEM_PROMPT = (
    "You are a fact-checker. For each statement, determine whether it is True or False. "
    "Use the available tools to search the internet or calculate. "
    "If you are not 100% certain and cannot find or calculate the answer, "
    "you MUST use the HumanInputRun tool to ask the user for the information — never guess."
    "Once you have a definitive answer, respond ONLY with 'True' or 'False'. "
    "Do not add any explanation or extra text."
)


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=API_KEY, model="gpt-5.4", base_url=BASE_URL, temperature=temperature
    )


def build_polygraph_agent() -> AgentExecutor:
    """
    Tool Calling Agent for binary fact-checking.

    Tools:
    - llm-math: evaluates math expressions
    - serpapi: web search for fact verification (optional, requires SERPAPI_API_KEY)
    - HumanInputRun: asks a human when the agent is uncertain
    """
    llm = get_llm()

    tool_names = ["llm-math"]
    if os.getenv("SERPAPI_API_KEY"):
        tool_names.append("serpapi")

    tools = load_tools(tool_names, llm=llm)
    tools.append(HumanInputRun())

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=5,
        handle_parsing_errors=True,
        verbose=True,
    )


def parse_bool(output: str) -> bool:
    """Parse agent output to Python bool. Defaults to True on parse failure."""
    return "false" not in output.lower()


def answer_questions(agent_executor: AgentExecutor) -> pd.DataFrame:
    """Download questions CSV, evaluate each statement, return DataFrame."""
    df = pd.read_csv(QUESTIONS_URL)
    statements = df["texts"].tolist()

    answers = []
    for statement in statements:
        print(f"\nStatement: {statement}")
        try:
            result = agent_executor.invoke({"input": statement})
            raw = result["output"]
        except Exception as e:
            print(f"  ERROR: {e} — defaulting to True")
            raw = "True"
        answer = parse_bool(raw)
        print(f"  Answer: {answer} (raw: {raw!r})")
        answers.append(answer)

    return pd.DataFrame({"texts": statements, "answers": answers})


def main():
    agent_executor = build_polygraph_agent()
    results = answer_questions(agent_executor)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSIONS_DIR / "m4agents_2_10_solution.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} answers to {output_path}")


if __name__ == "__main__":
    main()
