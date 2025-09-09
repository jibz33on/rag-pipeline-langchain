
# Retrieval-Augmented Generation (RAG) pipeline using LangChain, OpenAI, and Chroma

## Introduction

This notebook demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline step by step using LangChain, OpenAI embeddings, and ChromaDB.

In simple terms:

Imagine you want to ask questions about a website or document, but instead of manually reading it all, you let AI find the answers for you.

The AI (LLM) doesn’t just guess — it first retrieves the most relevant parts of the document and then uses those pieces to generate accurate answers.

This makes the model more reliable and context-aware, especially when working with private or domain-specific data.

## How It Works (Workflow)

### Load the Data

We fetch content from a webpage (Educosys GenAI course page in this demo).

The raw text becomes our knowledge source.

## Split into Chunks

Long text is broken down into smaller, overlapping chunks.

This helps the AI process information without losing context.

## Create Embeddings

Each text chunk is converted into a vector (numerical representation) using OpenAI embeddings.

Think of it as turning words into coordinates on a map.

## Store in Vector Database (ChromaDB)

All embeddings are stored in a vector database.

This allows fast similarity search when we ask questions later.

## Retrieve Relevant Chunks

When a user asks a question, the system searches the database for the most relevant text chunks.

Only the most useful context is passed to the AI model.


# Generate Answer with LLM

The retrieved context + user question is combined into a prompt.

The AI model (OpenAI Chat) then generates an answer grounded in the provided data.

      ┌─────────────────────┐
      │   Web / Documents   │   ← Source data (e.g., webpage, PDF, etc.)
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │   Split into Chunks │   ← Break text into smaller pieces
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │   Create Embeddings │   ← Convert text → numerical vectors
      │   (OpenAI model)    │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │   Vector Database   │   ← Store embeddings (ChromaDB)
      │   (Searchable)      │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │     Retriever       │   ← Find most relevant chunks for a query
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │   Prompt + Context  │   ← Combine user’s question + retrieved text
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │   LLM (OpenAI Chat) │   ← Generates grounded, contextual answer
      └─────────────────────┘

