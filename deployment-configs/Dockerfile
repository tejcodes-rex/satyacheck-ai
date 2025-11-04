# SatyaCheck AI - Enhanced Deepfake Detection Dockerfile
# Fast-build version: Models download on first run, then cached

FROM python:3.11-slim

# Set environment variables - Allow model downloads at runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    WORKERS=4 \
    THREADS=4 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and cache directories
WORKDIR /app
RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies with no cache to ensure fresh install
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD ALL DEEPFAKE DETECTION MODELS
# This eliminates HuggingFace rate limits and ensures fast cold starts
# Models are downloaded once during build, then cached in the Docker image
RUN python3 -c "import torch; \
from transformers import AutoImageProcessor, AutoModelForImageClassification, \
                         Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification; \
print('Pre-downloading ALL deepfake detection models (4 models total)...'); \
print('Downloading Image Deepfake Model (prithivMLmods/Deep-Fake-Detector-v2-Model)...'); \
AutoImageProcessor.from_pretrained('prithivMLmods/Deep-Fake-Detector-v2-Model'); \
AutoModelForImageClassification.from_pretrained('prithivMLmods/Deep-Fake-Detector-v2-Model'); \
print('Image model downloaded'); \
print('Downloading Video Deepfake Model (Wvolf/ViT_Deepfake_Detection)...'); \
AutoImageProcessor.from_pretrained('Wvolf/ViT_Deepfake_Detection'); \
AutoModelForImageClassification.from_pretrained('Wvolf/ViT_Deepfake_Detection'); \
print('Video model downloaded'); \
print('Downloading Audio Deepfake Model (MelodyMachine/Deepfake-audio-detection-V2)...'); \
Wav2Vec2FeatureExtractor.from_pretrained('MelodyMachine/Deepfake-audio-detection-V2'); \
Wav2Vec2ForSequenceClassification.from_pretrained('MelodyMachine/Deepfake-audio-detection-V2'); \
print('Audio primary model downloaded'); \
print('Downloading Audio Fallback Model (facebook/wav2vec2-base-960h)...'); \
Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h'); \
Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base-960h'); \
print('Audio fallback model downloaded'); \
print('All 4 models pre-downloaded and cached');"

# Copy application code (only essential files, test media excluded)
COPY *.py ./
COPY *.html ./
# Note: *.jpg, *.mp3, *.mp4 excluded by .dockerignore (test files only)

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

# Expose port
EXPOSE 8080

# Run with gunicorn for production
# Increased timeout to 600s for model loading on first request
CMD exec gunicorn --bind :$PORT \
    --workers $WORKERS \
    --threads $THREADS \
    --timeout 600 \
    --worker-class gthread \
    --access-logfile - \
    --error-logfile - \
    app:app
