# RAG Document Intelligence

Ask questions against your own PDFs. Upload a document, ask anything in plain English, get an answer with exact page citations — no hallucinations, no reaching outside the document.

Built with FastAPI + ChromaDB + HuggingFace MiniLM embeddings + Gemini 2.5 Flash, with a Streamlit frontend.

---

## How it works

1. A PDF is uploaded and text is extracted page-by-page with PyMuPDF.
2. Pages are split into overlapping chunks and embedded locally with `all-MiniLM-L6-v2`.
3. Embeddings are stored in ChromaDB under a named collection.
4. At query time, the question is embedded with the same model and the top-K most similar chunks are retrieved via cosine similarity.
5. Retrieved chunks are sent to Gemini 2.5 Flash with a strict system prompt that forces citation of page numbers and filenames.

---

## Stack

| Layer | Library |
|---|---|
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF parsing | PyMuPDF (fitz) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | ChromaDB (persistent, cosine distance) |
| LLM | Google Gemini 2.5 Flash via `google-genai` SDK |

---

## Prerequisites

- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/app/apikey) (free tier works)
- ~500 MB disk for the embedding model on first run

---

## Setup

```bash
git clone https://github.com/GurutejaReddy-04/rag-document-intelligence.git
cd rag-document-intelligence

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp env.example .env
# Edit .env and set GEMINI_API_KEY
```

---

## Running

Start the API and the frontend in separate terminals:

```bash
# Terminal 1 — backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## API reference

The FastAPI docs are available at `http://localhost:8000/docs` when the server is running.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Ingest a PDF into a named collection |
| `POST` | `/query` | Ask a question, get a cited answer |
| `GET` | `/collections` | List all collections |
| `DELETE` | `/collections/{name}` | Delete a collection |
| `POST` | `/reset` | Wipe all collections |
| `GET` | `/health` | Liveness check |

**Upload example:**

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@report.pdf" \
  -F "collection_name=q3-report" \
  -F "force=false"
```

**Query example:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the key risks?", "collection_name": "q3-report"}'
```

---

## Configuration

All options are set via environment variables (copy `env.example` to `.env`):

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required** |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Any Gemini model string |
| `CHROMA_PATH` | `chroma_db` | Where ChromaDB persists to disk |
| `UPLOAD_DIR` | `data/uploaded_docs` | Temp storage for uploads |
| `CHUNK_SIZE` | `500` | Characters per chunk; increase for dense technical docs |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `ALLOWED_ORIGINS` | `http://localhost:8501` | Comma-separated CORS origins |
| `API_URL` | `http://localhost:8000` | Backend URL used by the Streamlit frontend |

---

## Project structure

```
.
├── backend/
│   ├── config.py    # Environment variable loading
│   ├── db.py        # ChromaDB singleton client
│   ├── generator.py # Gemini prompt assembly and API call
│   ├── ingest.py    # PDF extraction, chunking, embedding, and storage
│   ├── main.py      # FastAPI app and route definitions
│   └── retriever.py # Similarity search against ChromaDB
├── frontend/
│   └── app.py       # Streamlit frontend
├── .gitignore
├── LICENSE
├── README.md
├── env.example
└── requirements.txt
```

---

## Collection naming rules

ChromaDB enforces constraints on collection names:

- 3 to 63 characters
- Letters, digits, hyphens (`-`), and underscores (`_`) only
- Must start and end with a letter or digit

Valid: `q3-report`, `ml_paper_2024`, `my-resume`  
Invalid: `_test`, `a`, `name with spaces`

---

## Re-ingesting a document

By default, uploading a file that already exists in a collection is a no-op (to prevent duplicates). To replace the existing chunks, set `force=true` in the upload form or API call.

---

## License

MIT
