from typing import List

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.core.config import settings


class CitedAnswer(BaseModel):
    answer: str = Field(
        description="Answer generated strictly from retrieved context."
    )
    citations: List[int] = Field(
        default_factory=list,
        description="Indices of retrieved chunks used."
    )


class GroqRAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0,
        )

    def ingest_documents(self, docs):
        """Create/update Chroma vector store."""
        return Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=settings.DATABASE_DIR,
        )

    def load_retriever(self, docs=None):
        """Load retriever from existing Chroma DB."""

        vectorstore = Chroma(
            persist_directory=settings.DATABASE_DIR,
            embedding_function=self.embeddings,
        )

        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        # Avoid BM25 crash when docs is empty
        if docs and len(docs) > 0:
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 3

            return EnsembleRetriever(
                retrievers=[
                    bm25_retriever,
                    vector_retriever,
                ],
                weights=[0.5, 0.5],
            )

        return vector_retriever

    @staticmethod
    def _format_context_with_indices(docs):
        formatted = []

        for idx, doc in enumerate(docs):
            formatted.append(
                f"--- Snippet Index [{idx}] ---\n{doc.page_content}"
            )

        return "\n\n".join(formatted)

    def get_execution_chain(self, retriever):

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a RAG assistant.

Rules:
- Answer only from the provided context.
- Do not invent facts.
- If the answer is not in the context, say:
  "I do not know based on the provided documents."

Context:
{context}
"""
                ),
                ("human", "{question}")
            ]
        )

        structured_llm = self.llm.with_structured_output(
            CitedAnswer
        )

        chain = (
            {
                "context": retriever | self._format_context_with_indices,
                "question": RunnablePassthrough(),
            }
            | prompt
            | structured_llm
        )

        return chain


rag_engine = GroqRAGEngine()