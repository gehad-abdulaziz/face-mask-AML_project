# ─────────────────────────────────────────────
# Face Mask Detection — Dockerfile
# Role 7: Deployer — Gehad
# ─────────────────────────────────────────────

FROM python:3.11-slim-bookworm

# Set working directory inside container
WORKDIR /app

# Fix Debian sources to use HTTPS instead of HTTP
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies (CPU-only torch)
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY api/ ./api/
COPY model/ ./model/
COPY static/ ./static/
# Expose port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]