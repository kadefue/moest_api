# Use a Python base image with common ML tools
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by LightGBM and XGBoost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]