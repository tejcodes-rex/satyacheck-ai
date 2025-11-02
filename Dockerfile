# SatyaCheck AI - Production Dockerfile
# Optimized for Google Cloud Run with scalability to 1M+ users

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    WORKERS=4 \
    THREADS=2

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

# Expose port
EXPOSE 8080

# Run with gunicorn for production (better than Flask dev server)
CMD exec gunicorn --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 --worker-class gthread --access-logfile - --error-logfile - app:app
