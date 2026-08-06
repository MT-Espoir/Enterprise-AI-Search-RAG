import sys
import os

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
