"""LangServe deployment — FastAPI server with a RAG agent and a joke chain."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes
from pydantic import BaseModel

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = "https://api.vsellm.ru/"
API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: Any


def cut_output(output):
    return output["output"]


# --- LLM & Embeddings ---
llm = ChatOpenAI(api_key=API_KEY, model="gpt-4o-mini", base_url=BASE_URL)
embeddings = OpenAIEmbeddings(api_key=API_KEY, model="text-embedding-3-small", base_url=BASE_URL)

# --- RAG Agent ---
data = WebBaseLoader("https://allopizza.su/spb/kupchino/about").load()
texts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(data)
db_embed = FAISS.from_documents(texts, embeddings)
retriever = db_embed.as_retriever()

retriever_tool = create_retriever_tool(retriever, "search_web", "Searches and returns data from page")

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, [retriever_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool])

# --- FastAPI app ---
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output) | cut_output,
    path="/rag_agent",
)

prompt2 = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(app, prompt2 | llm, path="/joke")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8501)
