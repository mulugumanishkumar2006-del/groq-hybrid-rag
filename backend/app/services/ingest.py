from typing import List, Dict, Any

from langchain_core.documents import Document

from app.core.engine import rag_engine


def process_and_ingest_raw_text(raw_documents: List[Dict[str, Any]]) -> dict:
    """
    Process incoming text and store it in ChromaDB.
    """

    if not raw_documents:
        return {
            "status": "error",
            "message": "No documents provided for ingestion."
        }

    try:
        langchain_docs = []

        for idx, item in enumerate(raw_documents):

            text_content = item.get("text", "").strip()

            if not text_content:
                continue

            metadata = {
                "source": "local_api_client",
                "chunk_index": idx,
            }

            langchain_docs.append(
                Document(
                    page_content=text_content,
                    metadata=metadata,
                )
            )

        if not langchain_docs:
            return {
                "status": "error",
                "message": "Zero valid text content extracted."
            }

        print(
            f"📦 Committing {len(langchain_docs)} documents to vector database..."
        )

        rag_engine.ingest_documents(langchain_docs)

        return {
            "status": "success",
            "count": len(langchain_docs),
            "message": f"Successfully indexed {len(langchain_docs)} chunks into disk storage."
        }

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")

        return {
            "status": "error",
            "message": str(e)
        }