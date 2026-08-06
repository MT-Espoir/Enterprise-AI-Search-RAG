import logging

from flask import Flask, send_from_directory, jsonify
from .config import config_map
from .extensions import jwt, mongo, cors, limiter, get_redis_client

# Import blueprints
from .api.auth import auth_bp
from .api.documents import documents_bp
from .api.chat import chat_bp
from .api.collections import collections_bp
from .api.health import health_bp
from .api.users import users_bp
from .api.dashboard import dashboard_bp

logger = logging.getLogger(__name__)


def create_app(config_name: str = "development") -> Flask:
    # Cấu hình thư mục static để phục vụ file HTML test
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    app.config.from_object(config_map[config_name])

    # ProxyFix — CHỈ khi chạy sau reverse proxy tin cậy (TRUST_PROXY=true). Cho
    # get_remote_address() (rate-limit key theo IP ở /login) đọc đúng client IP từ
    # X-Forwarded-For thay vì IP proxy (127.0.0.1). KHÔNG bật khi expose trực tiếp:
    # client sẽ spoof header để giả IP/vượt rate-limit. x_for=1 = tin đúng 1 lớp proxy.
    if app.config.get("TRUST_PROXY"):
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

    # Fail loud lúc khởi động nếu deploy production mà quên set secret thật —
    # KHÔNG được âm thầm chạy với SECRET_KEY/JWT_SECRET_KEY rỗng hoặc yếu (xem
    # config.py: ProductionConfig không có fallback, chỉ Dev/Testing mới có).
    if config_name == "production":
        missing = [k for k in ("SECRET_KEY", "JWT_SECRET_KEY") if not app.config.get(k)]
        if missing:
            raise RuntimeError(
                f"Thiếu biến môi trường bắt buộc cho production: {', '.join(missing)}. "
                f"Đặt giá trị ngẫu nhiên đủ mạnh (vd python -c \"import secrets; print(secrets.token_hex(32))\") "
                f"trước khi deploy — KHÔNG dùng giá trị mặc định/đoán được."
            )

        # FAIL LOUD: ChromaDB persistent mode dùng SQLite local, KHÔNG an toàn khi
        # nhiều gunicorn worker cùng ghi/đọc 1 thư mục -> "database is locked"/hỏng
        # dữ liệu. Chặn cứng ở production (đồng bộ triết lý fail-loud của secret
        # validation) — trừ khi CỐ TÌNH chạy 1 worker và xác nhận qua
        # ALLOW_SINGLE_WORKER_PERSISTENT=true. Multi-worker thật PHẢI CHROMA_MODE=http
        # (service Chroma riêng — đã có sẵn trong docker-compose.yml + nhánh
        # HttpClient trong vectorstore/client.py). Xem scale_architecture_exploration.md.
        if app.config.get("CHROMA_MODE") == "persistent" and not app.config.get("ALLOW_SINGLE_WORKER_PERSISTENT"):
            raise RuntimeError(
                "CHROMA_MODE=persistent (SQLite) KHÔNG an toàn với gunicorn nhiều worker "
                "(nhiều process ghi đồng thời -> hỏng dữ liệu). Đặt CHROMA_MODE=http cho "
                "multi-worker, HOẶC ALLOW_SINGLE_WORKER_PERSISTENT=true nếu CỐ TÌNH chạy đúng 1 worker."
            )
        if not app.config.get("BM25_SYNC_ENABLED"):
            logger.warning(
                "BM25_SYNC_ENABLED=false trong production: BM25 index sẽ RỖNG sau mỗi lần restart "
                "cho tới lần ingest kế tiếp, và KHÔNG đồng bộ giữa nhiều worker. Set BM25_SYNC_ENABLED=true."
            )

    @app.route('/')
    def serve_index():
        return send_from_directory(app.static_folder, 'index.html')

    # Khởi tạo extensions
    jwt.init_app(app)
    mongo.init_app(app)
    cors.init_app(app, origins=app.config["CORS_ALLOWED_ORIGINS"])

    # ── Rate limiting (Layer 6) ──────────────────────────────────────────────
    # RATELIMIT_ENABLED/RATELIMIT_STORAGE_URI được Flask-Limiter đọc trực tiếp từ
    # app.config lúc init_app. Storage: memory:// (dev 1 process) hoặc redis://
    # (multi-worker — đếm chung giữa worker). Giới hạn cụ thể gắn per-route bằng
    # @limiter.limit(...) trong từng blueprint (auth/chat/documents).
    limiter.init_app(app)

    # Trả lỗi 429 theo format JSON chuẩn của dự án (thay vì HTML mặc định).
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({
            "success": False,
            "error": {
                "message": "Bạn đã gửi quá nhiều yêu cầu trong thời gian ngắn. Vui lòng thử lại sau.",
                "code": "rate_limit_exceeded",
                "detail": str(e.description),
            }
        }), 429

    # ── Redis dùng chung + BM25 cross-worker sync (Layer 6) ──────────────────
    redis_client = get_redis_client(app.config.get("REDIS_URL"))
    if app.config.get("BM25_SYNC_ENABLED"):
        from .core.retrieval import start_bm25_sync_worker
        start_bm25_sync_worker(app, redis_client, app.config.get("BM25_SYNC_INTERVAL", 30))

    # Đăng ký blueprints
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(documents_bp, url_prefix="/api/documents")
    app.register_blueprint(chat_bp, url_prefix="/api/chat")
    app.register_blueprint(collections_bp, url_prefix="/api/collections")
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(users_bp, url_prefix="/api/users")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

    return app
