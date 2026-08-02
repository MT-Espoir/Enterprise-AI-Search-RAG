"""
BM25 cross-worker synchronization (Layer 6 — Production Hardening).

Vấn đề: BM25 index sống trong RAM của TỪNG process (get_bm25_index() là
singleton per-process). Hệ quả 2 tầng:

  1. (Ngay cả single-process) Một process VỪA khởi động có BM25 index RỖNG —
     HybridRetriever KHÔNG build lazily, và không hook nào build lúc startup.
     Hybrid search âm thầm suy biến thành vector-only cho tới khi có lần
     ingest/xóa đầu tiên trong process đó (bug tiềm ẩn: sau mỗi lần restart
     server, mất BM25 cho tới upload kế tiếp).

  2. (Multi-worker) Khi worker A ingest tài liệu mới, A refresh BM25 của riêng
     nó, nhưng worker B/C/D KHÔNG biết -> retrieval hybrid lệch nhau giữa các
     worker tùy user "trúng" worker nào (bug âm thầm, khó phát hiện qua benchmark).

Giải pháp (bật qua BM25_SYNC_ENABLED, xem config.py):
  - Mỗi worker khởi động 1 background thread:
      (a) BUILD ngay BM25 từ dữ liệu Chroma hiện có -> xử lý (1), mọi worker có
          index đầy đủ ngay sau khi boot (không cần chờ ingest).
      (b) Nếu có Redis: poll marker version (bm25:version). Worker nào ingest/xóa
          gọi mark_bm25_dirty() để INCR marker; các worker khác thấy version đổi
          -> tự refresh -> xử lý (2). Không có Redis (single-process): bỏ qua (b),
          vì ingest/xóa cục bộ đã refresh đồng bộ ngay trong pipeline/service.

Fail-open tuyệt đối: mọi lỗi Redis/Chroma trong thread này chỉ log warning, không
bao giờ làm chết request thật.
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

_BM25_VERSION_KEY = "bm25:version"


def mark_bm25_dirty(redis_client=None):
    """
    Bump marker version báo cho các worker khác biết BM25 corpus đã đổi (gọi
    SAU khi ingest/xóa đã refresh index cục bộ). Fail-open: không có Redis hoặc
    Redis lỗi -> bỏ qua im lặng (single-process không cần marker này).
    redis_client=None -> tự lấy singleton dùng chung qua get_redis_client().
    """
    if redis_client is None:
        from ..extensions import get_redis_client
        redis_client = get_redis_client()
    if redis_client is None:
        return
    try:
        redis_client.incr(_BM25_VERSION_KEY)
    except Exception as exc:
        logger.warning("mark_bm25_dirty thất bại (bỏ qua, fail-open): %s", exc)


def _build_index_once(app):
    """Build BM25 từ toàn bộ chunk Chroma hiện có. Cần app context (Chroma client
    đọc current_app.config)."""
    from .bm25_index import get_bm25_index
    from ..vectorstore.operations import VectorStoreOps
    with app.app_context():
        rebuilt = get_bm25_index().refresh(VectorStoreOps())
    return rebuilt


def start_bm25_sync_worker(app, redis_client, interval: int):
    """
    Khởi động background thread cho 1 worker. Trả về Thread (đã start) hoặc None.
    - Luôn build index 1 lần lúc khởi động (kể cả không có Redis).
    - Có Redis: sau build, vào vòng lặp poll marker version, refresh khi đổi.
    - Không Redis: chỉ build 1 lần rồi thoát (ingest/xóa cục bộ tự refresh đồng bộ).
    """

    def _run():
        # ĐỌC version TRƯỚC khi build (không phải sau) — nếu đọc sau, một ingest của
        # worker khác xảy ra TRONG lúc build sẽ INCR version, ta gán last_version =
        # giá trị đã tăng nhưng index vừa build lại KHÔNG chứa doc đó -> các poll sau
        # thấy current == last_version -> KHÔNG BAO GIỜ refresh cho thay đổi đó (bỏ
        # sót vĩnh viễn tới lần INCR kế tiếp). Đọc trước: worst case rebuild thừa 1
        # lần, KHÔNG bao giờ bỏ sót. (TOCTOU fix)
        try:
            last_version = redis_client.get(_BM25_VERSION_KEY) if redis_client is not None else None
        except Exception:
            last_version = None

        try:
            _build_index_once(app)
            logger.info("BM25 sync: build index lần đầu lúc khởi động xong.")
        except Exception as exc:
            logger.warning("BM25 sync: build index lần đầu lỗi (sẽ thử lại nếu có poll): %s", exc)

        if redis_client is None:
            logger.info("BM25 sync: không có Redis -> chỉ build 1 lần, không poll (single-process).")
            return

        while True:
            time.sleep(interval)
            try:
                current = redis_client.get(_BM25_VERSION_KEY)
                if current == last_version:
                    continue
                _build_index_once(app)
                last_version = current
                logger.info("BM25 sync: version đổi -> đã refresh index (version=%s).", current)
            except Exception as exc:
                logger.warning("BM25 sync: poll lỗi (thử lại sau %ss): %s", interval, exc)

    thread = threading.Thread(target=_run, daemon=True, name="bm25-sync")
    thread.start()
    logger.info("BM25 sync worker đã khởi động (interval=%ss, redis=%s).", interval, redis_client is not None)
    return thread
