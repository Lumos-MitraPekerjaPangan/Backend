FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install packages with special handling for problematic packages
RUN pip install --no-cache-dir Cython numpy
RUN pip install --no-cache-dir --no-build-isolation pystan 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=8080

# Command to run with Gunicorn for better performance
CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 --timeout 0 api:app
