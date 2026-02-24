FROM python:3.10-slim

# Install system dependencies for librosa (libsndfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY api.py .
COPY noise_classifier_model.h5 .
COPY .env* ./



# Run with gunicorn for production
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
