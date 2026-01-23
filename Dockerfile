# 1. Base image
FROM python:3.12-slim

# 2. Prevent Python buffering (better logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set workdir
WORKDIR /app

# 4. Install system deps (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy dependency list first (Docker cache)
COPY requirements.txt .

# 6. Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy app code
COPY . .

# 8. Expose port (Render expects 8000)
EXPOSE 8000

# 9. Start FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
