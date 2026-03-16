# Module 4: Retrieval-Augmented Generation (RAG)

Extend LLMs with external knowledge without fine-tuning or GPUs. Learn to load documents, chunk them into semantic pieces, convert to embeddings, store in vector databases, retrieve relevant chunks, and feed them into the model for grounded answers.

## What You'll Learn

- Why RAG is powerful: avoid hallucination, update knowledge without retraining
- Document loaders: ingest PDFs, web pages, text files
- Text splitters: chunk long documents into manageable pieces
- Embeddings: convert text to dense vectors for semantic search
- Vector stores: index and search embedding vectors (FAISS, Chroma)
- Retrievers: the abstraction layer between storage and the LLM
- RAG pipeline: orchestrating the full pipeline with LCEL

## What is RAG?

**Retrieval-Augmented Generation** solves a core LLM problem:

```
Problem: LLMs have fixed parametric knowledge (training cutoff date).
         They hallucinate when asked about new or private data.

Solution: Retrieve relevant documents at query time.
          Inject them into the prompt as context.
          The LLM answers based on the provided sources.
```

**Benefit**: Grounded, up-to-date answers without fine-tuning or GPU-intensive retraining.

## RAG Pipeline Overview

```
[1. Load]    [2. Split]    [3. Embed]    [4. Store]    [5. Retrieve]    [6. Generate]
Documents → Text Chunks → Vectors    → VectorDB  → Semantic Search → LLM + Prompt
   ↓            ↓             ↓            ↓             ↓                ↓
PDFs, web,  "sentence-     "Let me     Chroma,     Top-k chunks    "Answer this
text files  transformers"  embed you"  FAISS       based on query  using context"
```

---

## Stage 1: Document Loaders

Load external data into LangChain `Document` objects:

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Load a single text file
loader = TextLoader("path/to/file.txt")
documents = loader.load()
# Returns: [Document(page_content="...", metadata={"source": "..."}), ...]

# Load a PDF
loader = PyPDFLoader("path/to/file.pdf")
documents = loader.load()
# One Document per page, with metadata

# Load from web
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")
documents = loader.load()
```

**Document structure**:
```python
Document(
    page_content="The text of the document",
    metadata={"source": "file.txt", "page": 0}
)
```

---

## Stage 2: Text Splitters

Chunk long documents. Why? Context window limits and retrieval quality.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Characters per chunk
    chunk_overlap=200,      # Overlap for continuity
    separators=["\n\n", "\n", " ", ""]  # Try these first
)

chunks = splitter.split_documents(documents)
# Returns: [Document(...), Document(...), ...]
```

**Key parameters**:
- `chunk_size`: Smaller = more precise retrieval, more chunks to store. Larger = less storage, more context drift.
- `chunk_overlap`: Prevents information loss at chunk boundaries.
- `separators`: Respects document structure (paragraphs > sentences > words).

---

## Stage 3: Embeddings

Convert text to dense vectors for similarity search.

```python
from utils import OpenAIEmbeddings  # Course proxy

embeddings = OpenAIEmbeddings(api_key="your-course-key")

# Embed a single text
vector = embeddings.embed_query("What is photosynthesis?")
# Returns: [0.123, -0.456, ..., 0.789]  (768-dimensional vector)

# Embed multiple texts
vectors = embeddings.embed_documents([
    "Photosynthesis is how plants...",
    "Respiration is how cells..."
])
```

**Alternative: HuggingFace (Free, Open-Source)**

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query("What is photosynthesis?")
```

**Intuition**: Similar texts have similar vectors (high cosine similarity).

---

## Stage 4: Vector Store

Persist embeddings and index them for fast retrieval.

```python
from langchain_community.vectorstores import FAISS, Chroma
from utils import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key="your-key")

# Create and populate a vector store
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save to disk
vectorstore.save_local("my_vectorstore")

# Load later
vectorstore = FAISS.load_local("my_vectorstore", embeddings)
```

**Popular choices**:
- **FAISS** (Facebook): Fast, in-memory or disk-backed
- **Chroma**: Lightweight, persistent SQLite
- **Pinecone**: Cloud-hosted, fully managed
- **Weaviate**: Open-source, fully featured

---

## Stage 5: Retrievers

Query the vector store for relevant chunks.

```python
# Create a retriever from the vector store
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Return top 3 chunks
)

# Query
results = retriever.invoke("What is photosynthesis?")
# Returns: [Document(...), Document(...), Document(...)]
```

**Search types**:
- `similarity`: Nearest neighbours by cosine distance
- `similarity_score_threshold`: Only return if above threshold
- `mmr` (Maximal Marginal Relevance): Diverse results, not all identical

---

## Stage 6: The RAG Chain

Wire everything together with LCEL:

```python
from langchain import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from utils import ChatOpenAI

# Step 1: Load, split, embed, store
documents = loader.load()
chunks = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Step 2: Define the RAG prompt
template = """Use the following documents to answer the question.

Documents:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Step 3: Build the chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 4: Query
answer = rag_chain.invoke("What is photosynthesis?")
print(answer)
```

**Simpler version (all-in-one)**:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

rag_chain = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()
) | prompt | llm | StrOutputParser()

answer = rag_chain.invoke("What is photosynthesis?")
```

---

## Full Example: Build a QA System in 10 Lines

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from utils import ChatOpenAI, OpenAIEmbeddings

# Load, split, embed, store
docs = TextLoader("data.txt").load()
chunks = RecursiveCharacterTextSplitter().split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings(api_key="..."))

# Query
prompt = PromptTemplate.from_template("Answer based on:\n{context}\n\nQ: {question}\nA:")
chain = vectorstore.as_retriever() | prompt | ChatOpenAI() | StrOutputParser()
print(chain.invoke({"context": "...", "question": "What is RAG?"}))
```

---

## Notebooks in This Module

| Notebook | Topics |
|----------|--------|
| `M4_RAG.ipynb` | Document loaders, text splitters, embeddings, vector stores, retrievers, RAG pipeline |

---

## Debugging & Optimization

### Q: My retriever returns irrelevant chunks
- Try different `chunk_size` (smaller for precision, larger for context)
- Increase `k` (retrieve more chunks, rerank client-side)
- Switch to `mmr` search for diversity
- Check embedding model quality (use `sentence-transformers` for better semantics)

### Q: RAG is slow
- Use `similarity_score_threshold` to filter weak matches
- Cache embeddings if re-querying the same documents
- Use async loaders for parallel document loading
- Reduce `chunk_size` for fewer vectors to search

### Q: The model hallucinates despite retrieved context
- Add an explicit instruction: "Answer only using the provided documents. Say 'I don't know' if not found."
- Implement retrieval-based answer generation (e.g., use retrieved chunks as prompt, not as free-form context)

---

## Key Concepts

- **Semantic Search**: Finding similar text via vector similarity, not keyword matching
- **Embedding Dimension**: Larger (768, 1536) = more expressive but slower. Smaller (384) = faster but less nuanced.
- **Context Window**: Larger chunks = more context but longer processing. Balance with your model's limit.
- **Cold Start**: Retrieving before documents are indexed returns empty results. Always verify indexing succeeds.

## Further Reading

- [LangChain RAG Guide](https://python.langchain.com/docs/modules/data_connection/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers Models](https://www.sbert.net/)
