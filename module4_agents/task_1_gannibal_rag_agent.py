"""
Task 4.2 Step 6 — RAG Agent over Wikipedia article about Abram Gannibal.

Loads the Wikipedia page, builds a FAISS vector store, creates a retriever tool,
and uses a ReAct agent to answer questions from the dataset.
Output: step6_solution.csv with columns [question, answer], each answer ≤70 chars.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = "https://api.vsellm.ru/"
API_KEY = os.getenv("OPENAI_API_KEY")

WIKI_URL = "https://ru.wikipedia.org/wiki/%D0%93%D0%B0%D0%BD%D0%BD%D0%B8%D0%B1%D0%B0%D0%BB,_%D0%90%D0%B1%D1%80%D0%B0%D0%BC_%D0%9F%D0%B5%D1%82%D1%80%D0%BE%D0%B2%D0%B8%D1%87"
QUESTIONS_URL = "https://stepik.org/media/attachments/lesson/1107866/gannibal.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "submissions" / "step6_solution.csv"
FAISS_PATH = Path(__file__).resolve().parent / "gannibal_faiss_index"


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=API_KEY, model="gpt-4o-mini", base_url=BASE_URL, temperature=0.0
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=API_KEY, model="text-embedding-3-small", base_url=BASE_URL
    )


def build_agent():
    llm = get_llm()

    # Build or load vector store
    embeddings = get_embeddings()
    if FAISS_PATH.exists():
        print(f"Loading FAISS index from {FAISS_PATH}...")
        db_embed = FAISS.load_local(str(FAISS_PATH), embeddings, allow_dangerous_deserialization=True)
    else:
        print("Building FAISS index from Wikipedia...")
        data = WebBaseLoader(WIKI_URL).load()
        texts = CharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(data)
        db_embed = FAISS.from_documents(texts, embeddings)
        db_embed.save_local(str(FAISS_PATH))
        print(f"FAISS index saved to {FAISS_PATH}")
    retriever = db_embed.as_retriever(search_kwargs={"k": 6})

    # Retriever tool
    tool = create_retriever_tool(
        retriever,
        "search_gannibal",
        "Searches and returns information about Abram Petrovich Gannibal from Wikipedia",
    )

    # ReAct agent
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, [tool], prompt)
    return AgentExecutor(
        agent=agent, tools=[tool], verbose=True,
        max_iterations=15, handle_parsing_errors=True,
    )


def main():
    print("Loading questions...")
    df = pd.read_csv(QUESTIONS_URL)
    questions = df["question"].tolist()
    print(f"Got {len(questions)} questions")

    executor = build_agent()

    answers = []
    for i, q in enumerate(questions, 1):
        prompt = (
            f"{q}\n\n"
            "ВАЖНО: ответ должен быть максимально кратким, не более 70 символов."
        )
        try:
            result = executor.invoke({"input": prompt})
            answer = result["output"].strip()
        except Exception as e:
            print(f"  Error on question {i}: {e}")
            answer = ""

        # Trim to 70 chars as a hard limit
        if len(answer) > 70:
            answer = answer[:70]

        print(f"{i}. Q: {q}")
        print(f"   A: {answer!r} ({len(answer)} chars)")
        answers.append(answer)

    result_df = pd.DataFrame({"question": questions, "answer": answers})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(answers)} answers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()