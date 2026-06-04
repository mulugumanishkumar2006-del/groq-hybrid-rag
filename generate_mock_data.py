import os
import json


def setup_test_data():
    # Create directories
    raw_data_dir = os.path.join("data", "raw")
    processed_data_dir = os.path.join("data", "processed")

    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    print("📁 Created data directories")

    # Sample TXT file
    txt_content = """
Retrieval-Augmented Generation (RAG) Setup Guide

RAG is a technique that grants LLMs access to external knowledge bases.

The process involves three phases:

1. Ingestion
2. Retrieval
3. Generation

Ingestion:
Read documents, split into chunks, embed them, and store them in ChromaDB.

Retrieval:
Find the most relevant chunks based on the user query.

Generation:
Pass retrieved context and user query to the LLM.
"""

    txt_path = os.path.join(
        raw_data_dir,
        "rag_explanation.txt"
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content.strip())

    print(f"📄 Created: {txt_path}")

    # Sample JSON file
    json_content = [
        {
            "id": "doc_001",
            "title": "Company Remote Work Policy",
            "category": "HR",
            "content": (
                "Employees may work remotely up to "
                "3 days per week."
            ),
        },
        {
            "id": "doc_002",
            "title": "IT Password Requirements",
            "category": "Security",
            "content": (
                "Passwords must contain at least "
                "14 characters and be rotated every 90 days."
            ),
        },
    ]

    json_path = os.path.join(
        raw_data_dir,
        "company_policies.json"
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            json_content,
            f,
            indent=4
        )

    print(f"📄 Created: {json_path}")
    print("✅ Mock data generation completed")


if __name__ == "__main__":
    setup_test_data()