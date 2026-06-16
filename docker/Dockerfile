# ─────────────────────────────────────────────
# Face Mask Detection — Dockerfile (API)
# Role 7: Deployer — Gehad
# ─────────────────────────────────────────────

FROM python:3.11-slim-bookworm

WORKDIR /app

# Fix Debian sources
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-only torch + requirements
RUN pip install --no-cache-dir --timeout=300 \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy project files
COPY api/ ./api/
COPY model/mask_detector_v2.pth ./model/mask_detector_v2.pth
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]