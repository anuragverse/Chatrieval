FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed

# Run as a non-root user (uid 1000 also works on Hugging Face Spaces)
RUN useradd -m -u 1000 app
WORKDIR /home/app/backend

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so cold starts don't download it
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')" \
    && chown -R app:app /home/app

COPY --chown=app:app main.py .

USER app

# Sessions live in memory, so keep a SINGLE worker.
# Most hosts inject $PORT; Hugging Face Spaces uses 7860.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
