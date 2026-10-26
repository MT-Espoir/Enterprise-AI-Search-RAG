# ==============================================================
# Dockerfile — Backend (Flask + Gunicorn)
# ==============================================================
# Multi-stage build:
#   Stage 1 (builder): Cài dependencies vào virtual environment
#   Stage 2 (runner) : Copy venv + Tesseract (OCR tiếng Việt), chạy Gunicorn bằng user thường
#
# Model embedding/reranker (bge-m3, bge-reranker-v2-m3 ~6.5GB) KHÔNG đóng vào image:
# tải lần đầu vào HF_HOME=/models — mount volume vào đó để không tải lại mỗi lần tạo container.
# ==============================================================

# ---- Stage 1: Builder ----
FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build

# torch bản CPU trước: index mặc định trên Linux kéo bản CUDA (thêm vài GB thư viện nvidia)
# trong khi hệ thống chạy CPU-only. requirements.txt thấy torch đã thỏa nên không cài lại.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- Stage 2: Runner ----
FROM python:3.13-slim AS runner

# Tesseract + gói tiếng Việt cho OCR PDF scan (config.TESSERACT_CMD mặc định là đường dẫn Windows)
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    TESSERACT_CMD=/usr/bin/tesseract \
    UPLOAD_FOLDER=/app/uploads

WORKDIR /app
RUN mkdir -p /app/uploads /models && chown -R app:app /app /models
COPY --chown=app:app . .
USER app

EXPOSE 5000

# Mặc định 1 worker: mỗi worker nạp riêng bge-m3 + reranker (~5-6GB RAM). Tăng GUNICORN_WORKERS
# chỉ khi đủ RAM; luôn cần CHROMA_MODE=http + REDIS_URL + BM25_SYNC_ENABLED=true (xem docker-compose.yml).
# Timeout dài vì mỗi câu hỏi = 1 lần suy luận LLM local trên CPU. Không --preload: thread đồng bộ BM25
# được tạo trong create_app() phải chạy trong từng worker.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:5000 --worker-class gthread --workers ${GUNICORN_WORKERS:-1} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-600} --access-logfile - run:app"]
