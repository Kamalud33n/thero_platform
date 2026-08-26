FROM python:3.10.11-slim

# System libraries required by opencv-python / mediapipe at import & runtime
# (libGL, glib, X11 stubs) plus curl for the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime data directories (also volume-mounted in docker-compose.yml).
RUN mkdir -p data reports uploads

# App never opens a physical camera in a container.
ENV LOCAL_CAMERA_ENABLED=false \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
