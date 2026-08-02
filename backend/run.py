import sys
import os

# Console Windows mặc định dùng codepage (vd cp1252), không encode được ký tự
# có dấu tiếng Việt — crash structlog PrintLogger (request_tracer.py) khi log
# JSON câu hỏi/nội dung tiếng Việt (ensure_ascii=False). Reconfigure trước khi
# import bất kỳ module nào có thể ghi log.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from app import create_app

# Load các biến môi trường từ file .env
load_dotenv()

# Lấy môi trường hiện tại (mặc định là development)
env = os.getenv("FLASK_ENV", "development")
app = create_app(env)

if __name__ == "__main__":
    # Chỉ chạy debug server nếu môi trường là development
    is_debug = (env == "development")
    app.run(host="0.0.0.0", port=5000, debug=is_debug)
