FROM python:3.11-slim

WORKDIR /app

# Install minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch to fit in free tier
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from pathlib import Path; from sentence_transformers import SentenceTransformer; p = Path('./models/all-MiniLM-L6-v2'); m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); m.save(str(p))"

# Copy source code
COPY backend ./backend
COPY frontend ./frontend

# Expose default port
EXPOSE 8000
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
