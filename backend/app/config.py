import os
from datetime import timedelta

class BaseConfig:
    SECRET_KEY = os.getenv("SECRET_KEY")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/rag_db")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(seconds=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 3600)))
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(seconds=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES", 2592000)))
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    
    # Cấu hình Local LLM
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2:1b")
    
    CHROMA_MODE = os.getenv("CHROMA_MODE", "persistent")
    CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", os.path.join(os.path.dirname(__file__), "..", "chromadb_data"))
    CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
    CHROMA_TENANT = os.getenv("CHROMA_TENANT")
    CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", 5))
    RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 20))

    # Cấu hình Hybrid Retrieval (BM25 + Vector, RRF fusion)
    RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "hybrid")  # "vector" | "hybrid"
    HYBRID_VECTOR_POOL_SIZE = int(os.getenv("HYBRID_VECTOR_POOL_SIZE", 20))
    HYBRID_BM25_POOL_SIZE = int(os.getenv("HYBRID_BM25_POOL_SIZE", 20))
    HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", 60))

    # Query Processing Layer
    QUERY_PROCESSOR_SLM_ENABLED = os.getenv("QUERY_PROCESSOR_SLM_ENABLED", "true").lower() == "true"

    # Killswitch RIÊNG cho Decomposition/Multi-Query 
    DECOMPOSITION_ENABLED = os.getenv("DECOMPOSITION_ENABLED", "true").lower() == "true"

    # Minimal Observability — structured JSON request tracing (request_id, latency
    # breakdown, retrieval stats).
    OBSERVABILITY_ENABLED = os.getenv("OBSERVABILITY_ENABLED", "true").lower() == "true"

    # Tier 1 Security Guardrails (Input/Retrieval/Output) 
    INPUT_GUARDRAILS_ENABLED = os.getenv("INPUT_GUARDRAILS_ENABLED", "true").lower() == "true"
    RETRIEVAL_GUARDRAILS_ENABLED = os.getenv("RETRIEVAL_GUARDRAILS_ENABLED", "true").lower() == "true"
    OUTPUT_GUARDRAILS_ENABLED = os.getenv("OUTPUT_GUARDRAILS_ENABLED", "true").lower() == "true"

    # OCR cho PDF scan (Tesseract). TESSERACT_CMD mặc định trỏ vào đường dẫn cài đặt
    OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    # "vie" riêng, KHÔNG kèm "eng" 
    OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "vie")
    OCR_DPI = int(os.getenv("OCR_DPI", 200))
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(os.path.dirname(__file__), "..", "uploads"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50)) * 1024 * 1024

    # CORS — danh sách origin được phép
    CORS_ALLOWED_ORIGINS = [
        o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()
    ]

    # Redis + Rate limit + Multi-worker sync ──
    REDIS_URL = os.getenv("REDIS_URL", "")

    # Rate limiting (Flask-Limiter).
    RATELIMIT_ENABLED = os.getenv("RATELIMIT_ENABLED", "true").lower() == "true"
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", REDIS_URL or "memory://")
    RATELIMIT_LOGIN = os.getenv("RATELIMIT_LOGIN", "5 per minute")
    RATELIMIT_CHAT = os.getenv("RATELIMIT_CHAT", "20 per minute")
    RATELIMIT_UPLOAD = os.getenv("RATELIMIT_UPLOAD", "10 per minute")

    # BM25 cross-worker sync (xem app/core/bm25_sync.py).
    BM25_SYNC_ENABLED = os.getenv("BM25_SYNC_ENABLED", "false").lower() == "true"
    BM25_SYNC_INTERVAL = int(os.getenv("BM25_SYNC_INTERVAL", 30))  # giây giữa 2 lần poll

    # TRUST_PROXY: chỉ bật khi app CHẠY SAU reverse proxy tin cậy (nginx/gunicorn front).
    TRUST_PROXY = os.getenv("TRUST_PROXY", "false").lower() == "true"
    ALLOW_SINGLE_WORKER_PERSISTENT = os.getenv("ALLOW_SINGLE_WORKER_PERSISTENT", "false").lower() == "true"

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-insecure-secret-key-do-not-use-in-prod")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-only-insecure-jwt-secret-do-not-use-in-prod")

class ProductionConfig(BaseConfig):
    DEBUG = False
    # KHÔNG override SECRET_KEY/JWT_SECRET_KEY — kế thừa None từ BaseConfig
    # nếu thiếu env var, để create_app() raise RuntimeError lúc khởi động.

class TestingConfig(BaseConfig):
    TESTING = True
    MONGO_URI = "mongodb://localhost:27017/rag_test_db"
    SECRET_KEY = os.getenv("SECRET_KEY", "test-only-insecure-secret-key-do-not-use-in-prod")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "test-only-insecure-jwt-secret-do-not-use-in-prod")

config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
