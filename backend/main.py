import os
import uuid
from io import BytesIO
from typing import Annotated

import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Add it to backend/.env."
    )


# =========================================================
# CLIENTS / MODELS
# =========================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


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
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
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
#     "skipped_pages": [...]
# }
#
# Multiple PDFs are stored in ONE session/index.
# =========================================================

stores = {}


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    session_id: str
    question: str


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
        pages:
            [
                {
                    "text": "...",
                    "page": 1,
                    "source": "invoice.pdf"
                }
            ]

        skipped:
            [2, 5, 8]
    """

    reader = PdfReader(
        BytesIO(file_bytes)
    )

    pages = []
    skipped = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

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

        page_text = page["text"]

        split_texts = text_splitter.split_text(
            page_text
        )

        for chunk in split_texts:

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
    Generate local embeddings using MiniLM
    and create a FAISS cosine-similarity index.

    Since embeddings are normalized, inner product
    is equivalent to cosine similarity.
    """

    if not chunks:
        raise ValueError(
            "Cannot create FAISS index without chunks."
        )

    vectors = embedding_model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

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

    store = stores.get(session_id)

    if not store:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session not found. "
                "Upload and process your PDFs first."
            ),
        )

    if not question.strip():
        return []

    # -----------------------------------------------------
    # Embed user question
    # -----------------------------------------------------

    query_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    # -----------------------------------------------------
    # Search entire combined FAISS index
    # -----------------------------------------------------

    total_chunks = len(
        store["chunks"]
    )

    number_to_retrieve = min(
        top_k,
        total_chunks,
    )

    scores, indices = store["index"].search(
        query_vector,
        number_to_retrieve,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position < 0:
            continue

        results.append(
            {
                "text": store["chunks"][
                    index_position
                ],
                "metadata": store["metadata"][
                    index_position
                ],
                "score": float(score),
            }
        )

    return results


# =========================================================
# CONTEXT CREATION
# =========================================================

def create_context(results):
    """
    Create the context sent to Groq.

    Source and page metadata are included so the
    LLM knows where each piece of information came from.
    """

    context_parts = []

    for item in results:

        metadata = item["metadata"]

        source = metadata["source"]
        page = metadata["page"]

        context_parts.append(
            f"[Source: {source} | Page: {page}]\n"
            f"{item['text']}"
        )

    return "\n\n---\n\n".join(
        context_parts
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "Chatrieval API",
        "embedding_model": (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "vector_store": "FAISS",
        "llm": GROQ_MODEL,
    }


# =========================================================
# MULTI-PDF UPLOAD
# =========================================================

@app.post("/api/documents/upload")
async def upload_documents(
    files: Annotated[
        list[UploadFile],
        File(...)
    ]
):
    """
    Upload and process one or more PDFs.

    ALL readable PDFs are combined into a single
    FAISS index belonging to one session.

    Example:

        invoice1.pdf
        invoice2.pdf
        report.pdf

    become:

        One session
             |
             +-- invoice1 chunks
             +-- invoice2 chunks
             +-- report chunks
             |
             +-- One FAISS index
    """

    # -----------------------------------------------------
    # Validate upload
    # -----------------------------------------------------

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one PDF.",
        )

    # -----------------------------------------------------
    # Storage for ALL uploaded documents
    # -----------------------------------------------------

    all_pages = []
    skipped_pages = []
    document_names = []

    processed_documents = []
    empty_documents = []

    # -----------------------------------------------------
    # Process EVERY uploaded PDF
    # -----------------------------------------------------

    for file in files:

        filename = file.filename or ""

        # -------------------------------------------------
        # Validate extension
        # -------------------------------------------------

        if not filename.lower().endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{filename or 'Unknown file'} "
                    "is not a PDF."
                ),
            )

        # -------------------------------------------------
        # Read file
        # -------------------------------------------------

        content = await file.read()

        if not content:
            empty_documents.append(
                filename
            )
            continue

        # -------------------------------------------------
        # Extract text
        # -------------------------------------------------

        try:

            pages, skipped = extract_pdf_text(
                content,
                filename,
            )

        except Exception as exc:

            raise HTTPException(
                status_code=422,
                detail=(
                    f"Could not read "
                    f"{filename}: {exc}"
                ),
            )

        # -------------------------------------------------
        # Keep document name
        # -------------------------------------------------

        document_names.append(
            filename
        )

        # -------------------------------------------------
        # Store extracted pages
        # -------------------------------------------------

        if pages:

            processed_documents.append(
                filename
            )

            all_pages.extend(
                pages
            )

        else:

            empty_documents.append(
                filename
            )

        # -------------------------------------------------
        # Store skipped image-only pages
        # -------------------------------------------------

        skipped_pages.extend(
            {
                "source": filename,
                "page": page_number,
            }
            for page_number in skipped
        )

    # -----------------------------------------------------
    # Make sure readable text exists somewhere
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Build chunks from ALL PDFs
    # -----------------------------------------------------

    chunks, metadata = build_chunks(
        all_pages
    )

    if not chunks:

        raise HTTPException(
            status_code=422,
            detail=(
                "No usable text chunks were "
                "created from the uploaded PDFs."
            ),
        )

    # -----------------------------------------------------
    # Build ONE FAISS index containing ALL documents
    # -----------------------------------------------------

    try:

        index = build_faiss_index(
            chunks
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to create vector index: "
                f"{exc}"
            ),
        )

    # -----------------------------------------------------
    # Create session
    # -----------------------------------------------------

    session_id = str(
        uuid.uuid4()
    )

    # -----------------------------------------------------
    # Store session
    # -----------------------------------------------------

    stores[session_id] = {
        "index": index,
        "chunks": chunks,
        "metadata": metadata,
        "documents": document_names,
        "skipped_pages": skipped_pages,
    }

    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return {
        "session_id": session_id,

        "documents": document_names,

        "document_count": len(
            document_names
        ),

        "processed_documents": (
            processed_documents
        ),

        "empty_documents": (
            empty_documents
        ),

        "chunk_count": len(
            chunks
        ),

        "skipped_pages": (
            skipped_pages
        ),

        "message": (
            f"{len(document_names)} "
            "document(s) processed successfully."
        ),
    }


# =========================================================
# CHAT / QUESTION ANSWERING
# =========================================================

@app.post("/api/chat")
def chat(
    request: ChatRequest
):
    """
    Answer a question using retrieved chunks
    from ALL PDFs in the session.
    """

    question = request.question.strip()

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # -----------------------------------------------------
    # Retrieve relevant chunks
    # -----------------------------------------------------

    results = retrieve(
        request.session_id,
        question,
        top_k=5,
    )

    if not results:

        return {
            "answer": (
                "The uploaded documents do not "
                "contain enough information to "
                "answer this question."
            ),
            "sources": [],
        }

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = create_context(
        results
    )

    # -----------------------------------------------------
    # System prompt
    # -----------------------------------------------------

    system_prompt = """
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

    # -----------------------------------------------------
    # User prompt
    # -----------------------------------------------------

    user_prompt = (
        "DOCUMENT CONTEXT:\n\n"
        f"{context}"
        "\n\n"
        "USER QUESTION:\n\n"
        f"{question}"
    )

    # -----------------------------------------------------
    # Call Groq
    # -----------------------------------------------------

    try:

        completion = (
            groq_client.chat.completions.create(
                model=GROQ_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],

                temperature=0.2,

                max_tokens=2000,
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                f"LLM request failed: {exc}"
            ),
        )

    # -----------------------------------------------------
    # Extract answer
    # -----------------------------------------------------

    answer = (
        completion
        .choices[0]
        .message
        .content
        or ""
    )

    answer = answer.strip()

    # -----------------------------------------------------
    # Build unique sources
    # -----------------------------------------------------

    sources = []
    seen_sources = set()

    for item in results:

        metadata = item["metadata"]

        source = metadata["source"]
        page = metadata["page"]

        source_key = (
            source,
            page,
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(
            source_key
        )

        sources.append(
            {
                "source": source,
                "page": page,
                "score": round(
                    item["score"],
                    4,
                ),
            }
        )

    # -----------------------------------------------------
    # Return response
    # -----------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
    }