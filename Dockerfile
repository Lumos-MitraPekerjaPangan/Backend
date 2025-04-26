FROM python:latest

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=8080

# Command to run with Gunicorn for better performance
CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 --timeout 0 api:app
