# RAG Pipeline with LangChain

> A complete, runnable Retrieval-Augmented Generation pipeline from scratch — document loading through grounded answer generation, using LangChain, OpenAI, and ChromaDB.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings_+_Chat-412991?logo=openai)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-orange)](https://www.trychroma.com/)

---

## What This Is

A clean, step-by-step implementation of a RAG pipeline — the kind of system that's at the core of most production LLM applications. No abstractions hiding the important parts. Each stage is its own module so you can see exactly what's happening and where to plug in alternatives.

If you're learning RAG or evaluating LangChain for a project, this is a working reference implementation.

---

## The Pipeline

```
1. Load
   └── Documents from file/directory (PDF, TXT, web)
       LangChain DocumentLoaders

2. Chunk
   └── Split into overlapping text chunks
       RecursiveCharacterTextSplitter
       chunk_size=1000, chunk_overlap=200

3. Embed
   └── Convert chunks to dense vectors
       OpenAI text-embedding-ada-002

4. Store
   └── Persist vectors + metadata
       ChromaDB (local persistent collection)

5. Retrieve
   └── At query time: embed query → similarity search → top-k chunks
       ChromaDB similarity_search(query, k=4)

6. Generate
   └── Prompt: [system instructions] + [retrieved context] + [user question]
       OpenAI GPT-3.5-turbo / GPT-4
       Answer is grounded in retrieved documents
```

---

## Why Each Decision Was Made

**RecursiveCharacterTextSplitter with overlap:** Splitting at 1000 chars with 200-char overlap preserves sentence context at chunk boundaries. Pure fixed-size splitting without overlap loses context when a sentence straddles two chunks.

**text-embedding-ada-002:** The standard choice for production RAG — strong semantic quality, low cost, 1536 dimensions. Swap to `text-embedding-3-small` for higher throughput at slightly lower quality.

**ChromaDB:** Runs fully local with no infrastructure. For production scale, swap the vector store to Pinecone, Weaviate, or MongoDB Atlas — the LangChain VectorStore interface makes this a one-line change.

**Retrieval before generation:** The core RAG insight — instead of fine-tuning a model on your data (expensive, stale), keep documents in a retrieval index and inject relevant context at inference time. The model stays generic; the knowledge is in the store.

---

## Tech Stack

| Stage | Technology |
|-------|-----------|
| Document loading | LangChain DocumentLoaders |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector store | ChromaDB |
| LLM | OpenAI GPT-3.5-turbo / GPT-4 |
| Framework | LangChain (LCEL chain) |
| Language | Python 3.11 |

---

## Setup

```bash
git clone https://github.com/jibz33on/rag-pipeline-langchain
cd rag-pipeline-langchain

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

```bash
cp .env.example .env
# Add your OPENAI_API_KEY
```

---

## Usage

### Ingest documents

```python
from pipeline.ingest import ingest_documents

ingest_documents(
    source_dir="./docs",
    collection_name="my_docs"
)
# Loads, chunks, embeds, and stores to ChromaDB
```

### Query

```python
from pipeline.query import ask

answer = ask(
    question="What are the main themes in this document?",
    collection_name="my_docs"
)
print(answer)
```

### Full pipeline in one script

```bash
python run_pipeline.py \
  --docs ./docs \
  --question "Summarize the key points about X"
```

---

## Project Structure

```
rag-pipeline-langchain/
├── pipeline/
│   ├── ingest.py       # Load → chunk → embed → store
│   └── query.py        # Retrieve → prompt → generate
├── docs/               # Drop your source documents here
├── chroma_db/          # Persisted vector store (auto-created)
├── run_pipeline.py     # CLI entry point
├── .env.example
└── requirements.txt
```

---

## Extending This

**Swap the vector store:**
```python
# ChromaDB → Pinecone
from langchain.vectorstores import Pinecone
vectorstore = Pinecone.from_documents(docs, embeddings, index_name="my-index")
```

**Add reranking:**
After retrieval, add a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to reorder the top-k before passing to the LLM. Measurably improves answer quality on longer document sets.

**Add evaluation:**
Drop in RAGAS to measure faithfulness, answer relevancy, and context precision against a test set.

---

## Author

**Jibin Kunjumon** — AI Engineer  
[GitHub](https://github.com/jibz33on) · [LinkedIn](https://linkedin.com/in/jibin-kunjumon)
