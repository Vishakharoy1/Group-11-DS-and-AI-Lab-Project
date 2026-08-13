FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow / scipy / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (installed before requirements — see requirements.txt note)
COPY backend/requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Checkpoints + training artifacts (webapp/.dockerignore keeps only production models)
COPY output/ /app/output/

COPY backend/app/ /app/app/

EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
