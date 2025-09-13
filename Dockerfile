
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Configure apt to ignore GPG signature issues
RUN echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid-until && \
    echo 'APT::Get::AllowUnauthenticated "true";' > /etc/apt/apt.conf.d/99allow-unauth

# Install essential dependencies
RUN apt-get update --allow-releaseinfo-change --allow-unauthenticated && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    gcc \
    g++ \
    libmagic1 \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p FAQ \
    && mkdir -p logs \
    && mkdir -p user_data \
    && mkdir -p vector_store

# Expose port for WhatsApp Backend (includes WebSocket and HTML UI)
EXPOSE 8001

# Health check for WhatsApp Backend
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command to run WhatsApp Backend
CMD ["python", "start_whatsapp_backend.py", "--port", "8001", "--host", "0.0.0.0"]