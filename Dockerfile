# 1. Base image
FROM python:3.12-slim

# 2. Prevent Python buffering (better logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set workdir
WORKDIR /app

# 4. System deps
#    - build-essential: compiling Python packages
#    - libgomp1: OpenMP required by torch/numpy on slim
#    - libnss3 libnspr4 libgbm1 libglib2.0-0: required by kaleido (Plotly image export)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libnss3 \
    libnspr4 \
    libgbm1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy dependency list first (Docker cache)
COPY requirements.txt .

# 6. Install CPU-only torch first to avoid pulling in CUDA (~1.5 GB saved on Render)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 7. Install remaining Python deps (torch already satisfied, skipped)
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copy app code
COPY . .

# 9. Expose port (Render expects 8000)
EXPOSE 8000

# 10. Start FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
