# Chatrieval

Full-stack AI PDF assistant using React, Vite, FastAPI, local MiniLM embeddings, FAISS, and Groq GPT-OSS 120B.

## Architecture

```text
React + Vite
     |
     | REST API
     v
FastAPI
     |
     +--> PyPDF2 --> Chunking --> MiniLM --> FAISS
     |
     +--> Retrieved chunks --> Groq GPT-OSS 120B --> Answer + sources
```

OCR is intentionally disabled. Text-based pages are processed; image-only pages are skipped. A page containing normal text plus images still works—the text is extracted and the images are ignored.

## Prerequisites

- Python 3.10+
- Node.js 20.19+ (current Vite requirement)
- npm
- Groq API key

## Backend

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Start:

```powershell
uvicorn main:app --reload
```

API: `http://127.0.0.1:8000`
Docs: `http://127.0.0.1:8000/docs`

## Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Notes

The FAISS index is stored in backend memory for this local version. Restarting FastAPI clears the indexed documents. For deployment, add persistent storage/vector DB and authentication.
