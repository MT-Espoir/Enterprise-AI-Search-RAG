import logging

from flask_jwt_extended import JWTManager
from flask_pymongo import PyMongo
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logger = logging.getLogger(__name__)

jwt = JWTManager()
mongo = PyMongo()
cors = CORS()

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],  # không giới hạn toàn cục — chỉ áp per-route tường minh
    swallow_errors=True,
)


def user_or_ip_key():
    """Key rate-limit cho endpoint ĐÃ auth (chat/upload): ưu tiên JWT identity
    (per-user), fallback IP nếu chưa/không có token hợp lệ.

    Vì sao KHÔNG dùng IP cho các endpoint này: trong enterprise, mọi nhân viên
    thường sau CÙNG 1 IP egress công ty -> nếu key theo IP, cả công ty chung 1
    bucket (vd 20 chat/phút) -> user hợp lệ bị chặn nhầm, và 1 người có thể làm
    cạn quota của tất cả (DoS nội bộ). Key theo user giải quyết đúng.

    verify_jwt_in_request(optional=True): limiter đánh giá limit ở before_request
    — TRƯỚC khi @jwt_required trong view chạy — nên tại đây JWT chưa được verify;
    phải tự verify. optional=True -> không raise khi thiếu/sai token, chỉ trả về
    identity None để fallback IP (vd chính request bị 401 sau đó vẫn được đếm theo
    IP, không lọt lưới). login KHÔNG dùng key này (giữ IP — brute-force từ ngoài)."""
    from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
    try:
        verify_jwt_in_request(optional=True)
        identity = get_jwt_identity()
        if identity:
            return f"user:{identity}"
    except Exception:
        pass
    return get_remote_address()

# Redis client dùng chung (rate-limit storage + BM25 cross-worker sync marker).
# Fail-open tuyệt đối: dev/test không có Redis vẫn chạy bình thường (trả None).
_redis_client = None
_redis_initialized = False


def get_redis_client(redis_url: str = None):
    """
    Trả về redis client dùng chung (singleton per-process), hoặc None nếu không
    cấu hình REDIS_URL / không kết nối được. KHÔNG raise — caller phải xử lý None
    (fail-open). redis_url chỉ cần truyền ở lần gọi đầu (khởi tạo), các lần sau
    bỏ qua tham số.
    """
    global _redis_client, _redis_initialized
    if _redis_initialized:
        return _redis_client

    _redis_initialized = True
    if not redis_url:
        _redis_client = None
        return None

    try:
        import redis
        client = redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()  # xác nhận kết nối thật lúc khởi tạo (fail nhanh nếu Redis chết)
        _redis_client = client
        logger.info("Redis client đã kết nối: %s", redis_url)
    except Exception as exc:
        logger.warning("Không kết nối được Redis (%s) — chạy fail-open không có Redis: %s", redis_url, exc)
        _redis_client = None
    return _redis_client
