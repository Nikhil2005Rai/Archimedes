FROM python:3.11-slim

WORKDIR /app

# Install lightweight system dependencies (avoid build-essential to keep image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy python packaging configuration
COPY pyproject.toml .
COPY alembic.ini .

# Install dependencies without caching extra wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application source code & migrations
COPY app ./app
COPY migrations ./migrations

# Expose FastAPI port
EXPOSE 8000

# Default command
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
