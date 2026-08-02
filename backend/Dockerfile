# ==============================================================
# Dockerfile — Backend (Flask)
# ==============================================================
# Multi-stage build:
#   Stage 1 (builder): Cài đặt dependencies vào virtual environment
#   Stage 2 (runner) : Copy venv, chạy với Gunicorn
#
# TODO: Implement stages
# ==============================================================

# Stage 1: Builder
# FROM python:3.11-slim AS builder
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runner
# FROM python:3.11-slim AS runner
# WORKDIR /app
# COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
# COPY . .
# RUN mkdir -p uploads
# EXPOSE 5000
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "run:app"]
