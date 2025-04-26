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

# Install dependencies plus waitress (production WSGI server)
RUN pip install --no-cache-dir -r requirements.txt waitress

# No need for waitress in this case
# Copy application code
COPY . .

# Set environment variable for port
ENV PORT=8080

# Create a simple starter script
RUN echo 'import os\nfrom api import app\n\nif __name__ == "__main__":\n    port = int(os.environ.get("PORT", 8080))\n    app.run(host="0.0.0.0", port=port, debug=False)' > /app/server.py

# Run Flask directly
CMD python /app/server.py