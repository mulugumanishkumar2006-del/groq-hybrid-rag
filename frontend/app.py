import streamlit as st
import requests

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Grok Hybrid RAG Engine",
    layout="wide"
)

BACKEND_ENDPOINT = "http://127.0.0.1:8000"

# ----------------------------
# Header
# ----------------------------
st.title("⚡ Enterprise Knowledge Explorer")
st.write(
    "Production RAG application using ChromaDB, BM25 retrieval, and Groq LLMs."
)

# ----------------------------
# Sidebar - Document Ingestion
# ----------------------------
with st.sidebar:
    st.header("🗂️ Document Corpus Setup")

    raw_text_payload = st.text_area(
        "Paste documents (separate documents with a blank line)",
        value=(
            "The company insurance plan covers dental cleanings twice per year.\n\n"
            "For critical outages, contact oncall@enterprise.com.\n\n"
            "The annual code freeze begins on December 15."
        ),
        height=250
    )

    if st.button("📥 Ingest Documents", use_container_width=True):

        chunks = [
            chunk.strip()
            for chunk in raw_text_payload.split("\n\n")
            if chunk.strip()
        ]

        payload = {
            "documents": [
                {"text": chunk}
                for chunk in chunks
            ]
        }

        try:
            response = requests.post(
                f"{BACKEND_ENDPOINT}/ingest",
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()

                st.success(
                    data.get(
                        "message",
                        "Documents indexed successfully."
                    )
                )

            else:
                st.error(
                    f"Backend Error ({response.status_code})\n\n"
                    f"{response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI backend.\n\n"
                "Make sure the backend is running:\n"
                "uvicorn app.main:app --reload"
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")


# ----------------------------
# Query Section
# ----------------------------
st.subheader("🔍 Ask Questions")

user_question = st.text_input(
    "Ask a question about your indexed documents:"
)

if st.button("🚀 Query Documents", type="primary"):

    if not user_question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        response = requests.post(
            f"{BACKEND_ENDPOINT}/query",
            json={"question": user_question},
            timeout=120
        )

        if response.status_code == 200:

            data = response.json()

            st.markdown("## 🧠 Answer")
            st.success(data["answer"])

            st.markdown("### 📌 Citations")
            st.code(str(data.get("citations", [])))

            sources = data.get("sources", [])

            if sources:

                st.markdown("### 📄 Retrieved Sources")

                for source in sources:

                    with st.expander(
                        f"Chunk {source['snippet_idx']} | "
                        f"{source['source']}"
                    ):
                        st.write(source["content"])

        else:
            st.error(
                f"Backend Error ({response.status_code})\n\n"
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI backend.\n\n"
            "Start backend first:\n"
            "uvicorn app.main:app --reload"
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")