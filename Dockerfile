# SourceFidelity API – Dockerfile
FROM python:3.12.13-slim-bookworm@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579

WORKDIR /app

# Install system dependencies for PDF extraction, OCR, and controlled DOCX
# presentation rendering. The exact LibreOffice version is captured in each
# derivative's provenance; image rebuilds require rendered acceptance.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libreoffice-writer \
    fonts-dejavu-core \
    fonts-liberation \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash sourcefidelity && chown -R sourcefidelity:sourcefidelity /app
USER sourcefidelity

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')"

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
