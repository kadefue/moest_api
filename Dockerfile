FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for XGBoost/LightGBM
RUN apt-get update && apt-get install -y libgomp1 gcc g++

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]