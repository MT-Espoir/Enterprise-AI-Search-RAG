import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# Load .env TRƯỚC khi import app: app/config.py đọc os.getenv ngay lúc import (thuộc tính class). Import trước
# thì mọi biến .env về mặc định — trước đây chỉ đúng nhờ reloader của debug mode (tiến trình con kế thừa env).
load_dotenv()

from app import create_app  # noqa: E402

# Lấy môi trường hiện tại (mặc định là development)
env = os.getenv("FLASK_ENV", "development")
app = create_app(env)

if __name__ == "__main__":
    # Chỉ chạy debug server nếu môi trường là development
    is_debug = (env == "development")
    app.run(host="0.0.0.0", port=5000, debug=is_debug)
