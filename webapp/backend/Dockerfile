FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Pillow / scipy / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only PyTorch via extra index)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the checkpoint(s) and pre-computed result assets (CSVs, PNGs)
COPY output/ /app/output/

# Copy the backend application code
COPY backend/app/ /app/app/

# Render uses port 10000 by default for web services
EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
