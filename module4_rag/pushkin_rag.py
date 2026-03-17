import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATA_DIR = Path(__file__).resolve().parent / "pushkin_questions_data"
PDF_PATH = DATA_DIR / "The_Daughter_of_The_Commandant.pdf"
CSV_PATH = DATA_DIR / "pushkin_questions.csv"
INDEX_DIR = DATA_DIR / "pushkin_faiss_index"
OUTPUT_PATH = DATA_DIR / "pushkin_answers.csv"

BASE_URL = "https://api.vsellm.ru/"
API_KEY = os.getenv("OPENAI_API_KEY")


def load_and_split_pdf() -> list:
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(pages)


def get_or_create_vectorstore():
    embeddings = OpenAIEmbeddings(
        api_key=API_KEY, model="text-embedding-3-small", base_url=BASE_URL
    )

    if INDEX_DIR.exists():
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )

    print("Creating FAISS index from PDF...")
    chunks = load_and_split_pdf()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(INDEX_DIR))
    print(f"Saved index to {INDEX_DIR}")
    return db


def build_chain(db):
    llm = ChatOpenAI(
        api_key=API_KEY, model="gpt-4o-mini", base_url=BASE_URL, temperature=0.0
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        "Ответь на вопрос кратко (максимум 70 символов), используя только контекст.\n\n"
        "Контекст:\n{context}\n\n"
        "Вопрос: {question}\n"
        "Краткий ответ:"
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    db = get_or_create_vectorstore()
    chain = build_chain(db)

    df = pd.read_csv(CSV_PATH)
    questions = df["question"].tolist()

    answers = []
    for i, q in enumerate(questions, 1):
        answer = chain.invoke(q)
        print(f"{i}. Q: {q}")
        print(f"   A: {answer} ({len(answer)} chars)")
        answers.append(answer)

    result = pd.DataFrame({"question": questions, "answer": answers})
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(answers)} answers to {OUTPUT_PATH}")

    # Also save to submissions
    submissions_dir = Path(__file__).resolve().parent.parent / "submissions"
    submissions_path = submissions_dir / "pushkin_answers.csv"
    result.to_csv(submissions_path, index=False)
    print(f"Saved to {submissions_path}")


if __name__ == "__main__":
    main()
