"""Quản lý header ngữ cảnh của tài liệu: gợi ý, duyệt/sửa, ghi nhận lúc ingest (S2,
contextual_header_rollout_plan.md §2).

Mô hình trạng thái (trường trên MongoDB `documents`, xem models/document.py):
  - context_header            : bản ĐÃ DUYỆT (None = chưa duyệt)
  - context_header_suggested  : bản hệ thống gợi ý (rule | slm)
  - header HIỆU LỰC           : bản duyệt nếu có, ngược lại bản gợi ý (quyết định D1-B: dùng ngay
                                bản gợi ý để mọi tài liệu đều có header, tránh mất cân bằng kiểu q066)
  - context_header_version    : TĂNG mỗi khi header hiệu lực đổi chuỗi
  - embedded_header_version   : version mà embedding hiện tại phản ánh
Hai version lệch nhau ⇒ cần embed lại (job S6). Service này KHÔNG embed lại.
"""
import logging
from datetime import datetime, timezone

from bson import ObjectId

from ..core.context_header_suggest import HeaderSuggestion, normalize_header, ordered_texts, suggest_header
from ..extensions import mongo
from ..models.document import Document

logger = logging.getLogger(__name__)

SOURCE_USER = "user"
_UNSET = object()


def effective_header(doc: dict) -> str | None:
    return doc.get("context_header") or doc.get("context_header_suggested") or None


def plan_header_update(doc: dict, *, approved=_UNSET, suggested=_UNSET, suggested_source=_UNSET,
                       needs_review=_UNSET) -> dict:
    """Tính $set cho MongoDB khi đổi bản duyệt/gợi ý — hàm THUẦN để test không cần Mongo.

    Version chỉ tăng khi CHUỖI header hiệu lực đổi: duyệt nguyên văn bản gợi ý không đổi
    embedding nên không bắt embed lại.
    """
    new = dict(doc)
    if approved is not _UNSET:
        new["context_header"] = approved
    if suggested is not _UNSET:
        new["context_header_suggested"] = suggested
    if suggested_source is not _UNSET:
        new["context_header_suggested_source"] = suggested_source
    if needs_review is not _UNSET:
        new["context_header_needs_review"] = bool(needs_review)

    if new.get("context_header"):
        same_as_suggestion = new["context_header"] == new.get("context_header_suggested")
        source = new.get("context_header_suggested_source") if same_as_suggestion else SOURCE_USER
    else:
        source = new.get("context_header_suggested_source") if new.get("context_header_suggested") else None

    update = {
        "context_header": new.get("context_header"),
        "context_header_suggested": new.get("context_header_suggested"),
        "context_header_suggested_source": new.get("context_header_suggested_source"),
        "context_header_needs_review": bool(new.get("context_header_needs_review")),
        "context_header_source": source,
    }
    version = doc.get("context_header_version") or 0
    if effective_header(new) != effective_header(doc):
        version += 1
    update["context_header_version"] = version
    return update


def header_state(doc: dict) -> dict:
    """Biểu diễn trả về API."""
    version = doc.get("context_header_version") or 0
    embedded = doc.get("embedded_header_version") or 0
    if doc.get("context_header"):
        status = "approved"
    elif doc.get("context_header_suggested"):
        status = "suggested"
    else:
        status = "none"
    return {
        "doc_id": str(doc.get("_id")),
        "status": status,
        "effective": effective_header(doc),
        "approved": doc.get("context_header"),
        "suggested": doc.get("context_header_suggested"),
        "suggested_source": doc.get("context_header_suggested_source"),
        "source": doc.get("context_header_source"),
        "needs_review": bool(doc.get("context_header_needs_review")),
        "version": version,
        "embedded_version": embedded,
        "needs_reembed": version != embedded,
    }


class ContextHeaderService:
    def __init__(self, vector_ops=None, *, slm_enabled: bool = False, ollama_base_url: str = None,
                 slm_model: str = None):
        self.vector_ops = vector_ops
        self.slm_enabled = slm_enabled
        self.ollama_base_url = ollama_base_url or "http://localhost:11434"
        self.slm_model = slm_model or "qwen2.5:3b-instruct"

    @classmethod
    def from_config(cls, config, vector_ops=None) -> "ContextHeaderService":
        return cls(
            vector_ops,
            slm_enabled=config.get("CONTEXT_HEADER_SLM_ENABLED", True),
            ollama_base_url=config.get("OLLAMA_BASE_URL"),
            slm_model=config.get("LOCAL_LLM_MODEL"),
        )

    # ── Gợi ý ───────────────────────────────────────────────────────────────

    def suggest_from_chunks(self, filename: str, chunks: list[dict]) -> HeaderSuggestion:
        """chunks: [{"text", "metadata": {page_num, chunk_index}}] — dùng chung cho ingest lẫn tài liệu cũ."""
        return suggest_header(
            filename, ordered_texts(chunks), slm_enabled=self.slm_enabled,
            ollama_base_url=self.ollama_base_url, slm_model=self.slm_model,
        )

    def refresh_suggestion(self, doc_id: str) -> dict | None:
        """Gợi ý lại từ chunk đã lưu trong ChromaDB (tài liệu cũ không cần file gốc)."""
        doc = self._load(doc_id)
        if not doc:
            return None
        raw = self.vector_ops.collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        chunks = [{"text": t, "metadata": m} for t, m in zip(raw["documents"], raw["metadatas"])]
        if not chunks:
            raise ValueError("Tài liệu chưa có chunk nào (chưa ingest xong?) — không gợi ý được header.")
        suggestion = self.suggest_from_chunks(doc.get("original_name") or doc.get("filename") or "", chunks)
        return self._save(doc, plan_header_update(
            doc, suggested=suggestion.header, suggested_source=suggestion.source,
            needs_review=suggestion.needs_review,
        ))

    # ── Duyệt / sửa ─────────────────────────────────────────────────────────

    def set_approved(self, doc_id: str, header) -> dict | None:
        """header: chuỗi mới (được chuẩn hóa) hoặc None để bỏ bản duyệt (quay về dùng bản gợi ý)."""
        doc = self._load(doc_id)
        if not doc:
            return None
        if header is not None and not isinstance(header, str):
            raise ValueError("context_header phải là chuỗi hoặc null.")
        return self._save(doc, plan_header_update(doc, approved=normalize_header(header)))

    def approve_suggestion(self, doc_id: str) -> dict | None:
        doc = self._load(doc_id)
        if not doc:
            return None
        if not doc.get("context_header_suggested"):
            raise ValueError("Tài liệu chưa có header gợi ý để duyệt.")
        return self._save(doc, plan_header_update(doc, approved=doc["context_header_suggested"]))

    # ── Ghi nhận lúc ingest ─────────────────────────────────────────────────

    def record_ingest(self, doc_id: str, suggestion: HeaderSuggestion | None, embedded_header: str | None) -> None:
        """Lưu gợi ý của lần ingest + đánh dấu embedding đã phản ánh header hiệu lực hay chưa.
        Fail-open: lỗi ở đây không được làm hỏng ingest (chỉ log)."""
        try:
            doc = self._load(doc_id)
            if not doc:
                return
            update = plan_header_update(
                doc,
                suggested=suggestion.header if suggestion else None,
                suggested_source=suggestion.source if suggestion else None,
                needs_review=suggestion.needs_review if suggestion else True,
            )
            effective = update["context_header"] or update["context_header_suggested"]
            # Chunk vừa embed: có header hiệu lực → khớp version; không có header nào → cũng khớp
            # (không có gì để embed); có header nhưng KHÔNG embed (killswitch tắt) → lệch, chờ S6.
            if (embedded_header or None) == (effective or None):
                update["embedded_header_version"] = update["context_header_version"]
            self._save(doc, update)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{doc_id}] Không ghi được header gợi ý sau ingest ({exc}) — bỏ qua.")

    # ── Mongo ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(doc_id: str) -> dict | None:
        try:
            return mongo.db[Document.COLLECTION].find_one({"_id": ObjectId(doc_id)})
        except Exception:
            return None

    @staticmethod
    def _save(doc: dict, update: dict) -> dict:
        update = {**update, "updated_at": datetime.now(timezone.utc)}
        mongo.db[Document.COLLECTION].update_one({"_id": doc["_id"]}, {"$set": update})
        return header_state({**doc, **update})
