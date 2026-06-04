from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.documents import Document

from app.core.engine import rag_engine

app = FastAPI(title="Groq RAG Production API")

GLOBAL_RETRIEVER = None


def get_or_load_retriever():
    global GLOBAL_RETRIEVER

    if GLOBAL_RETRIEVER is None:
        try:
            GLOBAL_RETRIEVER = rag_engine.load_retriever()
            print("💾 Retriever restored from disk.")
        except Exception as e:
            print(f"Retriever load failed: {e}")
            GLOBAL_RETRIEVER = None

    return GLOBAL_RETRIEVER


class DocumentItem(BaseModel):
    text: str


class IngestRequest(BaseModel):
    documents: List[DocumentItem]


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[int] = []
    sources: List[Dict] = []


@app.post("/ingest")
async def ingest_documents(payload: IngestRequest):
    global GLOBAL_RETRIEVER

    try:
        docs = [
            Document(
                page_content=item.text,
                metadata={
                    "id": i,
                    "source": f"Doc_Chunk_{i}"
                }
            )
            for i, item in enumerate(payload.documents)
        ]

        rag_engine.ingest_documents(docs)

        GLOBAL_RETRIEVER = rag_engine.load_retriever(docs)

        return {
            "status": "success",
            "message": f"Successfully indexed {len(docs)} document fragments."
        }

    except Exception as e:
        print(f"Ingest Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/query", response_model=QueryResponse)
async def query_docs(payload: QueryRequest):

    retriever = get_or_load_retriever()

    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No documentation indices available. Please ingest documents first."
        )

    try:
        retrieved_docs = retriever.invoke(payload.question)

        chain = rag_engine.get_execution_chain(retriever)

        structured_response = chain.invoke(
            payload.question
        )

        sources_used = []

        for c_id in structured_response.citations:

            if c_id < len(retrieved_docs):

                sources_used.append(
                    {
                        "snippet_idx": c_id,
                        "source": retrieved_docs[c_id].metadata.get(
                            "source",
                            "Unknown"
                        ),
                        "content": retrieved_docs[c_id].page_content,
                    }
                )

        return QueryResponse(
            answer=structured_response.answer,
            citations=structured_response.citations,
            sources=sources_used,
        )

    except Exception as e:
        print(f"Query Error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )