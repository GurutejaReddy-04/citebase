"""
Streamlit frontend for the RAG Document Intelligence System.

Left sidebar  - upload PDFs, manage collections
Main panel    - select a collection, ask a question, read the answer + sources
"""

import os

import requests
import streamlit as st

# Set API_URL in .env to point at a non-localhost backend.
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- page config ---
st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG Document Intelligence System")
st.caption(
    "Upload any PDF and ask questions in plain English. "
    "Every answer is grounded in your document with exact page citations."
)

st.divider()


@st.cache_data(ttl=10)
def fetch_collections() -> list[str]:
    try:
        resp = requests.get(f"{API_URL}/collections", timeout=5)
        resp.raise_for_status()
        return resp.json().get("collections", [])
    except requests.exceptions.ConnectionError:
        return []


if "confirm_reset" not in st.session_state:
    st.session_state["confirm_reset"] = False


# --- sidebar: upload ---
with st.sidebar:
    st.header("Upload Document")
    st.markdown("Ingest a PDF into a named collection so you can query it later.")
    st.caption(
        "Collection names must be 3-63 characters and contain only letters, "
        "digits, hyphens (-), or underscores (_)."
    )

    collection_input = st.text_input(
        "Collection name",
        placeholder="e.g. ml-paper, q3-report",
    )
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    force_ingest  = st.checkbox("Force re-ingestion (replace existing chunks)")

    if st.button("Ingest Document", use_container_width=True):
        if not uploaded_file:
            st.warning("Please select a PDF file first.")
        elif not collection_input.strip():
            st.warning("Please enter a collection name.")
        else:
            with st.spinner("Extracting text, chunking, and embedding..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                        data={
                            "collection_name": collection_input.strip(),
                            "force": force_ingest,
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    st.success(response.json()["message"])
                    st.cache_data.clear()
                except requests.exceptions.HTTPError as e:
                    st.error(f"Upload failed: {e.response.json().get('detail', str(e))}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the backend. Is it running on port 8000?")

    st.divider()

    # --- sidebar: manage collections ---
    st.subheader("Manage Collections")
    st.caption("Delete a single collection or reset the entire database.")

    collections = fetch_collections()

    if collections:
        col_to_delete = st.selectbox(
            "Choose collection to delete",
            options=collections,
            key="del_collection",
        )
        if st.button("🗑️ Delete Selected Collection", use_container_width=True):
            with st.spinner("Deleting collection..."):
                try:
                    resp = requests.delete(
                        f"{API_URL}/collections/{col_to_delete}", timeout=10
                    )
                    resp.raise_for_status()
                    st.success(resp.json()["message"])
                    st.cache_data.clear()
                    st.rerun()
                except requests.exceptions.HTTPError as e:
                    st.error(e.response.json().get("detail", str(e)))
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the backend.")
    else:
        st.info("No collections exist yet.")

    if st.button("💣 Wipe ALL Collections", use_container_width=True):
        st.session_state["confirm_reset"] = True

    if st.session_state.get("confirm_reset"):
        confirm = st.checkbox("Yes, I understand this will delete every collection permanently.")
        if confirm and st.button("Confirm Reset", type="primary", use_container_width=True):
            with st.spinner("Resetting the system..."):
                try:
                    resp = requests.post(f"{API_URL}/reset?wipe_uploads=false", timeout=10)
                    resp.raise_for_status()
                    st.success(resp.json()["message"])
                    st.cache_data.clear()
                    st.session_state["confirm_reset"] = False
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the backend.")
                except Exception as e:
                    st.error(f"Reset failed: {e}")
        elif not confirm:
            st.warning("Please check the box above to proceed.")

    st.divider()
    st.caption("Backend: FastAPI · ChromaDB · HuggingFace MiniLM\nLLM: Gemini 2.5 Flash")


# --- main panel: query ---
collections = fetch_collections()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Select Collection")
    if collections:
        selected_collection = st.selectbox(
            "Available collections",
            options=collections,
            label_visibility="collapsed",
        )
    else:
        st.info("No collections found. Upload a document first.")
        selected_collection = None

with col_right:
    st.subheader("Ask a Question")
    question = st.text_area(
        "Your question",
        placeholder="What are the key findings of this paper?",
        label_visibility="collapsed",
        height=100,
    )

    ask_btn = st.button(
        "Get Answer",
        disabled=(not selected_collection or not question.strip()),
        use_container_width=True,
        type="primary",
    )

st.divider()

if ask_btn and question.strip() and selected_collection:
    with st.spinner("Searching document and generating answer..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question":        question.strip(),
                    "collection_name": selected_collection,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            st.subheader("Answer")
            st.markdown(data["answer"])

            if data.get("sources"):
                st.subheader("Sources used")
                for src in data["sources"]:
                    # Score is cosine distance (0-2); convert to a 0-100% similarity.
                    similarity = max(0, 1 - src["score"] / 2) * 100
                    st.markdown(
                        f"📄 **{src['source']}** — Page {src['page']}  "
                        f"`relevance: {similarity:.0f}%`"
                    )

        except requests.exceptions.HTTPError as e:
            st.error(f"Query failed: {e.response.json().get('detail', str(e))}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the backend. Is it running on port 8000?")
