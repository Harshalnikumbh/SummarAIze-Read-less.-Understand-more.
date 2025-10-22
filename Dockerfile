# Use an official lightweight Python image
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port (Railway and Hugging Face use PORT env)
EXPOSE 7860

# Define environment variable
ENV PORT=7860

# Command to run your app (replace app.py with your file)
CMD ["python", "app.py"]
