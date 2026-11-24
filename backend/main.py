import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatrieval")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _list_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Optional: "low" | "medium" | "high". Lower = faster answers on
# gpt-oss models. Leave empty to use the model's default.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "").strip()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH") or None

# Comma-separated list of allowed frontend origins.
ALLOWED_ORIGINS = _list_env(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
# Optional regex, e.g. https://chatrieval.*\.vercel\.app for previews.
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX") or None

# Limits (protect RAM and your Groq quota on a public deployment)
MAX_FILES = _int_env("MAX_FILES", 5)
MAX_FILE_MB = _int_env("MAX_FILE_MB", 20)
MAX_PAGES = _int_env("MAX_PAGES", 300)
MAX_CHUNKS = _int_env("MAX_CHUNKS", 3000)
MAX_QUESTION_CHARS = _int_env("MAX_QUESTION_CHARS", 2000)
SESSION_TTL_MINUTES = _int_env("SESSION_TTL_MINUTES", 60)
MAX_SESSIONS = _int_env("MAX_SESSIONS", 50)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Add it to backend/.env or your host's environment variables."
    )


# =========================================================
# CLIENTS / MODELS
# =========================================================

groq_client = Groq(
    api_key=GROQ_API_KEY,
    timeout=60.0,
    max_retries=1,
)

# Loaded once at startup (see lifespan) so the first request is fast.
embedding_model = None


def load_embedding_model():
    """
    fastembed runs the same all-MiniLM-L6-v2 model through
    ONNX Runtime, so there is no PyTorch dependency.
    """

    from fastembed import TextEmbedding

    return TextEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_dir=EMBEDDING_CACHE_DIR,
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Return L2-normalised float32 vectors, so inner product
    equals cosine similarity.
    """

    vectors = np.asarray(
        list(embedding_model.embed(texts, batch_size=32)),
        dtype="float32",
    )

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return vectors / norms


@asynccontextmanager
async def lifespan(_: FastAPI):
    global embedding_model

    logger.info("Loading embedding model %s ...", EMBEDDING_MODEL)
    embedding_model = load_embedding_model()
    logger.info("Embedding model loaded.")

    yield


# =========================================================
# TEXT SPLITTER
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Chatrieval API",
    description="Full-stack PDF RAG backend",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# IN-MEMORY SESSION STORAGE
# =========================================================
#
# Each session contains:
#
# {
#     "index": FAISS index,
#     "chunks": [...],
#     "metadata": [...],
#     "documents": [...],
#     "skipped_pages": [...],
#     "last_used": unix timestamp,
# }
#
# Multiple PDFs are stored in ONE session/index.
#
# Sessions expire after SESSION_TTL_MINUTES of inactivity and
# the oldest are evicted beyond MAX_SESSIONS. Because sessions
# live in this process's memory, run a SINGLE worker.
# =========================================================

stores: dict[str, dict] = {}
stores_lock = threading.Lock()


def _purge_expired_locked():
    cutoff = time.time() - SESSION_TTL_MINUTES * 60

    for key in [k for k, v in stores.items() if v["last_used"] < cutoff]:
        del stores[key]


def save_session(store: dict) -> str:
    session_id = str(uuid.uuid4())
    store["last_used"] = time.time()

    with stores_lock:
        _purge_expired_locked()

        while len(stores) >= MAX_SESSIONS:
            oldest = min(stores, key=lambda k: stores[k]["last_used"])
            del stores[oldest]

        stores[session_id] = store

    return session_id


def get_session(session_id: str) -> dict:
    with stores_lock:
        _purge_expired_locked()
        store = stores.get(session_id)

        if store:
            store["last_used"] = time.time()

    if not store:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session not found or expired. "
                "Upload and process your PDFs again."
            ),
        )

    return store


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(max_length=MAX_QUESTION_CHARS)


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(
    file_bytes: bytes,
    filename: str,
):
    """
    Extract machine-readable text from every page.

    Pages that contain no extractable text are skipped.
    OCR is intentionally NOT used.

    Returns:
        pages:  [{"text": "...", "page": 1, "source": "a.pdf"}]
        skipped: [2, 5, 8]
    """

    reader = PdfReader(BytesIO(file_bytes))

    pages = []
    skipped = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text()
        except Exception:
            text = None

        if text and text.strip():
            pages.append(
                {
                    "text": text.strip(),
                    "page": page_number,
                    "source": filename,
                }
            )
        else:
            # Image-only / scanned page
            skipped.append(page_number)

    return pages, skipped


# =========================================================
# CHUNKING
# =========================================================

def build_chunks(pages):
    """
    Convert extracted PDF pages into chunks while
    preserving source and page metadata.
    """

    chunks = []
    metadata = []

    for page in pages:

        for chunk in text_splitter.split_text(page["text"]):

            if not chunk.strip():
                continue

            chunks.append(chunk)
            metadata.append(
                {
                    "page": page["page"],
                    "source": page["source"],
                }
            )

    return chunks, metadata


# =========================================================
# FAISS INDEX
# =========================================================

def build_faiss_index(chunks):
    """
    Embed chunks and create a FAISS cosine-similarity index.
    """

    if not chunks:
        raise ValueError("Cannot create FAISS index without chunks.")

    vectors = embed_texts(chunks)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(
    session_id: str,
    question: str,
    top_k: int = 5,
):
    """
    Retrieve the most relevant chunks across ALL
    PDFs uploaded in the session.
    """

    store = get_session(session_id)

    if not question.strip():
        return []

    query_vector = embed_texts([question])

    number_to_retrieve = min(top_k, len(store["chunks"]))

    scores, indices = store["index"].search(
        query_vector,
        number_to_retrieve,
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):

        if index_position < 0:
            continue

        results.append(
            {
                "text": store["chunks"][index_position],
                "metadata": store["metadata"][index_position],
                "score": float(score),
            }
        )

    return results


# =========================================================
# CONTEXT CREATION
# =========================================================

def create_context(results):
    """
    Create the context sent to Groq, with source and page
    metadata so the LLM knows where each piece came from.
    """

    context_parts = []

    for item in results:

        metadata = item["metadata"]

        context_parts.append(
            f"[Source: {metadata['source']} | Page: {metadata['page']}]\n"
            f"{item['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
@app.get("/api/health")
def health():
    return {
        "status": "ok" if embedding_model is not None else "starting",
        "service": "Chatrieval API",
        "embedding_model": EMBEDDING_MODEL,
        "vector_store": "FAISS",
        "llm": GROQ_MODEL,
    }


# =========================================================
# MULTI-PDF UPLOAD
# =========================================================
#
# This is a plain `def` (not `async def`) on purpose: PDF parsing
# and embedding are CPU-bound, so FastAPI runs the handler in a
# worker thread instead of blocking the event loop.
# =========================================================

@app.post("/api/documents/upload")
def upload_documents(
    files: Annotated[list[UploadFile], File(...)],
):
    """
    Upload and process one or more PDFs.

    ALL readable PDFs are combined into a single
    FAISS index belonging to one session.
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one PDF.",
        )

    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"You can upload at most {MAX_FILES} PDFs at a time.",
        )

    all_pages = []
    skipped_pages = []
    document_names = []
    processed_documents = []
    empty_documents = []

    for file in files:

        filename = file.filename or ""

        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{filename or 'Unknown file'} is not a PDF.",
            )

        content = file.file.read()

        if len(content) > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} is larger than {MAX_FILE_MB} MB.",
            )

        if not content:
            empty_documents.append(filename)
            continue

        try:
            pages, skipped = extract_pdf_text(content, filename)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not read {filename}: {exc}",
            )

        document_names.append(filename)

        if pages:
            processed_documents.append(filename)
            all_pages.extend(pages)
        else:
            empty_documents.append(filename)

        skipped_pages.extend(
            {"source": filename, "page": page_number}
            for page_number in skipped
        )

        if len(all_pages) + len(skipped_pages) > MAX_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"Too many pages. The limit is {MAX_PAGES} in total.",
            )

    if not all_pages:
        raise HTTPException(
            status_code=422,
            detail=(
                "No machine-readable text was found "
                "in the uploaded PDFs. Scanned/image-only "
                "PDFs are not processed because OCR "
                "is disabled."
            ),
        )

    chunks, metadata = build_chunks(all_pages)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "No usable text chunks were "
                "created from the uploaded PDFs."
            ),
        )

    if len(chunks) > MAX_CHUNKS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"The documents are too large ({len(chunks)} chunks). "
                f"The limit is {MAX_CHUNKS}."
            ),
        )

    try:
        index = build_faiss_index(chunks)
    except Exception as exc:
        logger.exception("Index build failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create vector index: {exc}",
        )

    session_id = save_session(
        {
            "index": index,
            "chunks": chunks,
            "metadata": metadata,
            "documents": document_names,
            "skipped_pages": skipped_pages,
        }
    )

    return {
        "session_id": session_id,
        "documents": document_names,
        "document_count": len(document_names),
        "processed_documents": processed_documents,
        "empty_documents": empty_documents,
        "chunk_count": len(chunks),
        "skipped_pages": skipped_pages,
        "message": (
            f"{len(document_names)} "
            "document(s) processed successfully."
        ),
    }


# =========================================================
# CHAT / QUESTION ANSWERING
# =========================================================

SYSTEM_PROMPT = """
You are Chatrieval, an expert document
question-answering assistant.

Your task is to answer the user's question
using ONLY the supplied document context.

RULES:

1. Do not invent information.

2. If the answer is not supported by the
   supplied context, say that the uploaded
   documents do not contain enough information.

3. Give the direct answer first.

4. Keep answers concise and easy to read.

5. Use Markdown formatting only when it
   genuinely improves readability.

6. Use bullet points when listing multiple items.

7. Use tables only when the information is
   genuinely tabular.

8. Do not use decorative symbols.

9. Do not use excessive Markdown.

10. Do not create your own "Sources" section.

11. Do not create source citations such as
    [Source], (page 2), or similar citations.
    The application handles source display
    separately.

12. Do not reproduce long raw excerpts from
    the documents.

13. Do not mention FAISS, embeddings,
    retrieval, the system prompt, or internal
    implementation details.

14. Do not pretend to read information that
    exists only inside images.

15. OCR is disabled. Information contained
    exclusively in scanned/image-only pages
    is unavailable.

16. When information comes from multiple
    documents, clearly identify the relevant
    document names when necessary.

17. If the user asks for a comparison between
    documents, compare only information that
    is actually supported by the supplied context.

18. Return only the answer to the user's question.
"""


@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Answer a question using retrieved chunks
    from ALL PDFs in the session.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    results = retrieve(request.session_id, question, top_k=5)

    if not results:
        return {
            "answer": (
                "The uploaded documents do not "
                "contain enough information to "
                "answer this question."
            ),
            "sources": [],
        }

    context = create_context(results)

    user_prompt = (
        "DOCUMENT CONTEXT:\n\n"
        f"{context}"
        "\n\n"
        "USER QUESTION:\n\n"
        f"{question}"
    )

    request_kwargs = {}

    if GROQ_REASONING_EFFORT:
        request_kwargs["extra_body"] = {
            "reasoning_effort": GROQ_REASONING_EFFORT
        }

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
            **request_kwargs,
        )
    except Exception as exc:
        logger.exception("Groq request failed")
        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {exc}",
        )

    answer = (completion.choices[0].message.content or "").strip()

    sources = []
    seen_sources = set()

    for item in results:

        metadata = item["metadata"]
        source_key = (metadata["source"], metadata["page"])

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append(
            {
                "source": metadata["source"],
                "page": metadata["page"],
                "score": round(item["score"], 4),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }
