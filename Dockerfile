# Dockerfile

# Use a standard, slim Python base image for a stable environment
FROM python:3.12-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install all system dependencies required by yt-dlp for full impersonation functionality
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    brotli \
    openssl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python dependencies, including the crucial curl_cffi extra
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the application when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
