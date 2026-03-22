# Module 4 — Retrieval-Augmented Generation (RAG)

Extend LLMs with external knowledge without fine-tuning: load documents, chunk them, convert to embeddings, store in a vector database, retrieve relevant chunks, and feed them into the model for grounded answers.

---

## Topics Covered

- Why RAG: avoid hallucination, update knowledge without retraining
- Document loaders — ingest PDFs, web pages, text files via LangChain
- Text splitters — chunk long documents with `RecursiveCharacterTextSplitter`
- Embeddings — convert text to dense vectors for semantic search
- Vector stores — index and search with FAISS and Chroma
- Retrievers — abstraction layer between storage and the LLM
- Full RAG pipeline — orchestrated with LCEL
- Custom text splitter — section-based chunking for structured documents

---

## Lecture Notes

### M4_RAG.ipynb

**Document loaders** — LangChain adapters that read files or web pages and emit a list of `Document` objects.
`PyPDFLoader(path).load()` returns one `Document` per PDF page; each has `.page_content` (text) and `.metadata` (source, page number). The metadata is preserved through the entire pipeline and surfaced in retrieval results to show provenance.
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
docs = loader.load()   # list of Document objects
print(docs[0].metadata)   # {'source': 'document.pdf', 'page': 0}
```

**TextSplitter** — Divides long documents into overlapping chunks that fit within the model's context window.
`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` splits on `\n\n`, then `\n`, then space, trying to preserve paragraph boundaries. The `chunk_overlap` parameter repeats the last `n` characters in the next chunk to avoid cutting a sentence mid-thought; higher overlap means more chunks and more retrieval calls.
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(len(chunks), "chunks from", len(docs), "pages")
```

**Embeddings + FAISS** — Converts text chunks to dense vectors and builds a searchable in-memory index.
`OpenAIEmbeddings()` encodes each chunk as a float vector; `FAISS.from_documents(chunks, embeddings)` builds the index in one call. Persist with `index.save_local("path")` and reload with `FAISS.load_local(...)` — this avoids re-embedding on each run, which is the expensive step.
```python
from utils import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
embeddings = OpenAIEmbeddings()
index = FAISS.from_documents(chunks, embeddings)
index.save_local("my_index")
# Reload later
index = FAISS.load_local("my_index", embeddings, allow_dangerous_deserialization=True)
```

**LCEL RAG pipeline** — Composes retriever and LLM into a single runnable using the `|` pipe operator.
`{"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()` passes the query to the retriever and the original text to the prompt simultaneously. `RunnablePassthrough()` is the key primitive — it threads the unmodified question alongside the retrieved context so both appear in the prompt template.
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
answer = rag_chain.invoke("What happened in chapter 3?")
```

**MMR** — Maximal Marginal Relevance — retrieval strategy that balances similarity to the query with diversity among results.
`vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})` fetches `fetch_k` candidates by cosine similarity, then greedily selects `k` that are most different from each other. The non-obvious reason to use MMR: plain top-k can return five nearly identical chunks from the same paragraph — MMR spreads coverage across the document.
```python
retriever = index.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
docs = retriever.get_relevant_documents("Who is Pugachev?")
```

---

## Files

| File | Description |
|------|-------------|
| `M4_RAG.ipynb` | Document loaders, text splitters, embeddings, vector stores, retrievers, RAG pipeline |
| `custom_text_splitter.py` | Custom section-based text splitter — parses numbered headings and extracts content per section |
| `pushkin_rag.py` | RAG pipeline over "The Captain's Daughter" PDF — answers literature questions using FAISS + OpenAI embeddings |
| `pushkin_questions_data/` | PDF source and question CSV for the Pushkin RAG exercise |

---

## How to Run

```bash
pip install langchain langchain-community langchain-openai faiss-cpu pypdf sentence-transformers
jupyter notebook module4_rag/M4_RAG.ipynb

# Run the Pushkin RAG pipeline
python module4_rag/pushkin_rag.py
```

---

## Key Concepts

- **RAG (Retrieval-Augmented Generation)**: Retrieve relevant documents at query time and inject them into the prompt — grounds the model in up-to-date or private data without fine-tuning.
- **Embedding**: A dense vector representation of text; similar texts have high cosine similarity.
- **Semantic Search**: Finding similar text via vector similarity, not keyword matching.
- **`chunk_size` / `chunk_overlap`**: Smaller chunks = more precise retrieval; overlap prevents information loss at chunk boundaries.
- **FAISS**: Facebook's fast, in-memory vector index; can be saved/loaded from disk with `save_local` / `load_local`.
- **Retriever**: The interface between a vector store and the LLM chain — returns top-k documents for a query.
- **MMR (Maximal Marginal Relevance)**: Retrieval strategy that maximises result diversity, not just similarity.
- **`RunnablePassthrough`**: LCEL primitive that passes the input unchanged — used to thread the question alongside retrieved context.

---

## References

- [LangChain RAG Guide](https://python.langchain.com/docs/modules/data_connection/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
