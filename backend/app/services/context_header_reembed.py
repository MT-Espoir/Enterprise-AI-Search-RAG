"""Job embed lại chunk theo header ngữ cảnh HIỆU LỰC (S6, contextual_header_rollout_plan.md §3).

Vì sao cần: duyệt/sửa header (S2–S4) chỉ đổi MongoDB → context_header_version lệch
embedded_header_version. Job này đưa embedding về khớp:
  - đọc chunk từ ChromaDB (KHÔNG cần file gốc — tài liệu cũ đã mất file);
  - embedding = build_context_text(header hiệu lực, nội dung gốc); documents GIỮ NGUYÊN;
    metadata["context_header"] = header (None → xóa khóa, Chroma gộp metadata khi update);
  - chunk_id giữ nguyên → `collection.update` tại chỗ, nhãn benchmark vẫn dùng được;
  - so CHUỖI header từng chunk: chunk đã mang đúng header thì bỏ qua (chạy lại = làm tiếp,
    và tài liệu "lệch version nhưng cùng chuỗi" chỉ cần đánh dấu, không embed lại);
  - ghi embedded_header_version SAU khi xong từng tài liệu, có điều kiện version không đổi
    trong lúc chạy (header bị sửa giữa chừng → để lệch, lần chạy sau xử lý);
  - xong toàn bộ → BM25 refresh 1 lần + mark_bm25_dirty() cho worker khác.

Job luôn ghép header vào embedding, KHÔNG phụ thuộc killswitch CONTEXT_HEADER_ENABLED:
killswitch chỉ quyết định BM25/Reranker/Generator có đọc header hay không (bật SAU khi
job xong — S7, tránh mất cân bằng tạm thời kiểu q066).

Chạy TRONG tiến trình server (§3b): Chroma persistent = SQLite, script riêng ghi song song
là 2 tiến trình cùng ghi. Trạng thái + khóa nằm ở MongoDB (`jobs`) để mọi worker cùng thấy
tiến độ và không 2 worker chạy cùng lúc.
"""
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from ..core.context_header import build_context_text
from ..extensions import mongo
from ..models.document import Document
from .context_header_service import effective_header

logger = logging.getLogger(__name__)

JOBS_COLLECTION = "jobs"
JOB_ID = "context_header_reembed"
DEFAULT_BATCH_SIZE = 16
# Không có heartbeat quá lâu = tiến trình chạy job đã chết (server tắt giữa chừng)
STALE_AFTER = timedelta(minutes=10)


class JobAlreadyRunning(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    # PyMongo trả datetime không kèm múi giờ (UTC) — so sánh với _now() cần aware
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def chunks_to_update(target_header: str | None, metadatas: list[dict]) -> list[int]:
    """Vị trí các chunk có embedding CHƯA phản ánh target_header — hàm THUẦN.
    Bất biến S1: metadata["context_header"] chỉ có khi embedding thật sự chứa header đó."""
    target = (target_header or "").strip()
    return [i for i, meta in enumerate(metadatas) if ((meta or {}).get("context_header") or "").strip() != target]


def documents_needing_reembed(doc_ids: list[str] | None = None) -> list[dict]:
    """Tài liệu đã ingest xong có version lệch; doc_ids → chỉ xét các tài liệu đó (vẫn bỏ qua
    tài liệu không lệch — chunk tự quyết qua so chuỗi)."""
    query = {"status": Document.STATUS_DONE}
    if doc_ids:
        query["_id"] = {"$in": [ObjectId(i) for i in doc_ids]}
    else:
        query["$expr"] = {"$ne": [{"$ifNull": ["$context_header_version", 0]},
                                  {"$ifNull": ["$embedded_header_version", 0]}]}
    projection = {"filename": 1, "original_name": 1, "context_header": 1, "context_header_suggested": 1,
                  "context_header_version": 1, "embedded_header_version": 1}
    return list(mongo.db[Document.COLLECTION].find(query, projection).sort("_id", 1))


class ReembedJobStore:
    """Trạng thái job trên MongoDB — 1 document `_id = JOB_ID`, đồng thời là khóa."""

    @property
    def _coll(self):
        return mongo.db[JOBS_COLLECTION]

    def get(self) -> dict | None:
        state = self._coll.find_one({"_id": JOB_ID})
        if not state:
            return None
        state = {k: v for k, v in state.items() if k != "_id"}
        heartbeat = _aware(state.get("heartbeat_at"))
        if state.get("status") == "running" and heartbeat and _now() - heartbeat > STALE_AFTER:
            state["status"] = "interrupted"  # tiến trình chạy job đã chết — cho phép chạy lại
        for key, value in list(state.items()):
            if isinstance(value, datetime):
                state[key] = _aware(value).isoformat()
        return state

    def acquire(self, initial: dict) -> bool:
        """Nhận khóa nguyên tử: chỉ khi chưa có job hoặc job cũ không còn chạy / đã chết."""
        now = _now()
        fresh = {**initial, "status": "running", "started_at": now, "heartbeat_at": now, "finished_at": None,
                 "cancel_requested": False, "error": None}
        taken = self._coll.find_one_and_update(
            {"_id": JOB_ID, "$or": [{"status": {"$ne": "running"}}, {"heartbeat_at": {"$lt": now - STALE_AFTER}}]},
            {"$set": fresh}, return_document=ReturnDocument.AFTER,
        )
        if taken:
            return True
        try:
            self._coll.insert_one({"_id": JOB_ID, **fresh})
            return True
        except DuplicateKeyError:
            return False

    def update(self, fields: dict) -> dict:
        """Ghi tiến độ + heartbeat; trả về trạng thái hiện tại (để đọc cancel_requested)."""
        return self._coll.find_one_and_update(
            {"_id": JOB_ID}, {"$set": {**fields, "heartbeat_at": _now()}}, return_document=ReturnDocument.AFTER,
        ) or {}

    def request_cancel(self) -> bool:
        return self._coll.update_one({"_id": JOB_ID, "status": "running"},
                                     {"$set": {"cancel_requested": True}}).modified_count == 1

    def finish(self, status: str, error: str | None = None, **fields) -> None:
        self._coll.update_one({"_id": JOB_ID}, {"$set": {
            **fields, "status": status, "error": error, "finished_at": _now(), "heartbeat_at": _now(),
        }})


class ContextHeaderReembedder:
    def __init__(self, vector_ops, embedder, bm25_index=None, batch_size: int = DEFAULT_BATCH_SIZE,
                 store: ReembedJobStore | None = None):
        self.vector_ops = vector_ops
        self.embedder = embedder
        self.bm25_index = bm25_index
        self.batch_size = max(1, int(batch_size))
        self.store = store or ReembedJobStore()

    # ── Xem trước (không ghi gì) ────────────────────────────────────────────

    def preview(self, doc_ids: list[str] | None = None) -> dict:
        documents, total_chunks, mark_only = [], 0, 0
        for doc in documents_needing_reembed(doc_ids):
            metas = self._chunks(str(doc["_id"]), include=["metadatas"])["metadatas"]
            pending = len(chunks_to_update(effective_header(doc), metas))
            total_chunks += pending
            mark_only += pending == 0
            documents.append({"doc_id": str(doc["_id"]), "filename": doc.get("original_name") or doc.get("filename"),
                              "header": effective_header(doc), "chunks": len(metas), "chunks_to_embed": pending})
        return {"documents": len(documents), "chunks_to_embed": total_chunks,
                "documents_mark_only": mark_only, "items": documents}

    # ── Chạy ────────────────────────────────────────────────────────────────

    def start_async(self, doc_ids: list[str] | None = None, started_by: str | None = None) -> threading.Thread:
        documents = documents_needing_reembed(doc_ids)
        if not self.store.acquire({
            "started_by": started_by, "params": {"doc_ids": doc_ids, "batch_size": self.batch_size},
            "total_docs": len(documents), "processed_docs": 0, "embedded_docs": 0, "mark_only_docs": 0,
            "failed_docs": [], "embedded_chunks": 0, "current": None, "chunks_per_second": None,
            "bm25_refreshed": False,
        }):
            raise JobAlreadyRunning("Job embed lại header đang chạy.")
        thread = threading.Thread(target=self._run, args=(documents,), daemon=True, name="context-header-reembed")
        thread.start()
        logger.info(f"Bắt đầu embed lại header ngữ cảnh: {len(documents)} tài liệu")
        return thread

    def _run(self, documents: list[dict]) -> None:
        processed = embedded_docs = mark_only = embedded_chunks = 0
        failed: list[dict] = []
        t0 = time.perf_counter()
        try:
            for doc in documents:
                doc_id = str(doc["_id"])
                filename = doc.get("original_name") or doc.get("filename")
                state = self.store.update({"current": {"doc_id": doc_id, "filename": filename}})
                if state.get("cancel_requested"):
                    break
                try:
                    count, cancelled = self._reembed_document(doc, t0, embedded_chunks)
                except Exception as exc:  # noqa: BLE001 — 1 tài liệu lỗi không dừng cả job
                    logger.error(f"[{doc_id}] Embed lại header thất bại: {exc}", exc_info=True)
                    failed.append({"doc_id": doc_id, "filename": filename, "error": str(exc)})
                    continue
                embedded_chunks += count
                if cancelled:
                    break
                processed += 1
                embedded_docs += count > 0
                mark_only += count == 0
                self.store.update({"processed_docs": processed, "embedded_docs": embedded_docs,
                                   "mark_only_docs": mark_only, "embedded_chunks": embedded_chunks,
                                   "failed_docs": failed})
            cancelled = bool(self.store.update({"current": None}).get("cancel_requested"))
            bm25_refreshed = self._refresh_bm25() if embedded_chunks else False
            self.store.finish("cancelled" if cancelled else "done", processed_docs=processed,
                              embedded_docs=embedded_docs, mark_only_docs=mark_only,
                              embedded_chunks=embedded_chunks, failed_docs=failed, bm25_refreshed=bm25_refreshed)
            logger.info(f"Embed lại header xong: {processed} tài liệu, {embedded_chunks} chunk, lỗi {len(failed)}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Job embed lại header dừng vì lỗi: {exc}", exc_info=True)
            self.store.finish("failed", error=str(exc), processed_docs=processed, embedded_chunks=embedded_chunks,
                              failed_docs=failed)

    def _reembed_document(self, doc: dict, t0: float, embedded_before: int) -> tuple[int, bool]:
        """Trả về (số chunk đã embed, có bị hủy giữa chừng không)."""
        doc_id = str(doc["_id"])
        header = effective_header(doc)
        version = doc.get("context_header_version") or 0
        raw = self._chunks(doc_id, include=["documents", "metadatas"])
        todo = chunks_to_update(header, raw["metadatas"])
        done = 0
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start:start + self.batch_size]
            texts = [build_context_text(header, raw["documents"][i]) for i in batch]
            self.vector_ops.collection.update(
                ids=[raw["ids"][i] for i in batch],
                embeddings=self.embedder.embed_batch(texts),
                metadatas=[{"context_header": header or None} for _ in batch],
            )
            done += len(batch)
            elapsed = time.perf_counter() - t0
            state = self.store.update({
                "embedded_chunks": embedded_before + done,
                "chunks_per_second": round((embedded_before + done) / elapsed, 3) if elapsed else None,
            })
            if state.get("cancel_requested"):
                return done, True  # chunk đã ghi giữ nguyên header mới; lần sau so chuỗi làm tiếp
        # Đánh dấu có điều kiện: header bị sửa trong lúc chạy → version đã tăng → không khớp → để lệch
        version_filter = version if version else {"$in": [0, None]}
        mongo.db[Document.COLLECTION].update_one(
            {"_id": doc["_id"], "context_header_version": version_filter},
            {"$set": {"embedded_header_version": version, "updated_at": _now()}},
        )
        return done, False

    def _chunks(self, doc_id: str, include: list[str]) -> dict:
        return self.vector_ops.collection.get(where={"doc_id": doc_id}, include=include)

    def _refresh_bm25(self) -> bool:
        if self.bm25_index is None:
            return False
        self.bm25_index.refresh(self.vector_ops)
        from ..core.retrieval import mark_bm25_dirty
        mark_bm25_dirty()
        return True
