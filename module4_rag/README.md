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
