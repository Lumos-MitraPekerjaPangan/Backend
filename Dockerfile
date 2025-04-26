# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install build dependencies required for Prophet
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable for port
ENV PORT=8080

# Create a prestart script to check if Firebase authentication works
RUN echo '#!/bin/bash\necho "Starting app pre-check..."\npython -c "import firebase_admin; print(\"Firebase SDK imported successfully\")"' > /app/prestart.sh && \
    chmod +x /app/prestart.sh

# Run the prestart check and then start Gunicorn with additional parameters
CMD /app/prestart.sh && exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 300 --preload --log-level debug api:app