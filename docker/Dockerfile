FROM python:3.11-slim

# System dependencies required by Pillow/OpenCV on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (includes src/models/best_model.pth)
COPY src/ src/

# Imports resolve correctly: api.py, model.py, preprocessing.py, train.py are all under src/
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Build: docker build -f docker/Dockerfile -t cifar10-api .
# Run:   docker run -p 8000:8000 cifar10-api
