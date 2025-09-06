# Multi-service Docker image combining transcription, LLM, and TTS services
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Update package lists and install all system dependencies
RUN apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libasound2-dev \
    ffmpeg \
    build-essential \
    gcc \
    espeak-ng \
    alsa-utils \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY app.py .

# Create output directory for audio files
RUN mkdir -p /app/output

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports for potential web APIs
EXPOSE 8000

# Default command - you can override this when running the container
# Example: docker run myimage python transcribe_realtime.py
# Example: docker run myimage python llm_wrapper.py  
# Example: docker run myimage python tts_app.py
CMD ["python", "app.py"]
