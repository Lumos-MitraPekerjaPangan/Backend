# --- Base Image ---
    FROM python:3.12-slim

    ENV authentication="0AkvpaHdXSP4enx"
    
    # --- Set work directory ---
    WORKDIR /app
    
    # --- Install system dependencies ---
    RUN apt-get update && apt-get install -y \
        gcc \
        build-essential \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        libxml2-dev \
        libxslt1-dev \
        libjpeg-dev \
        zlib1g-dev \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*
    
    # --- Install Prophet specific dependencies ---
    RUN pip install --upgrade pip
    RUN pip install --no-cache-dir prophet
    
    # --- Copy project files ---
    COPY . .
    
    # --- Install Python dependencies ---
    RUN pip install --no-cache-dir -r requirements.txt
    
    # --- Expose port ---
    EXPOSE 5000
    
    # --- Command to run the app ---
    CMD ["python", "api.py"]
    