# Minimal RAG LLM Demo

This repo contains a small Retrieval Augmented Generation (RAG) example.

## What it does

- Loads PDF documents from `data/pdfs`
- Splits them into chunks using `RecursiveCharacterTextSplitter`
- Builds a Chroma vector store with `text-embedding-3-small`
- Uses `gpt-4o-mini` via `ChatOpenAI` to answer questions grounded in those documents

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
