# Groq Hybrid RAG — Document Q&A System

A Retrieval-Augmented Generation application for question-answering over 
custom documents, combining semantic (vector) and keyword (BM25) retrieval 
for more accurate, grounded, citation-backed answers.

## Status
🚧 Functional locally. Not yet deployed.

## Key Features
- Hybrid retrieval: ChromaDB vector search + BM25 keyword search
- Groq-hosted LLM for fast response generation
- FastAPI backend + Streamlit frontend
- Citation-aware responses grounded only in retrieved context

## How It Works
Documents are embedded (Hugging Face embeddings) and indexed in ChromaDB. 
Queries run through the hybrid retrieval pipeline, relevant chunks are 
passed to a Groq model, and the response is returned with source citations.

## Tech Stack
Python, FastAPI, Streamlit, LangChain, ChromaDB, Hugging Face Embeddings, 
BM25, Groq API

## Planned Next
PDF/Word ingestion, reranking, multi-user auth, cloud deployment
