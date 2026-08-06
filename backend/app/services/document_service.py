import hashlib
import logging
import os
import re
from datetime import datetime, timezone

from ..models.document import Document
from ..models.department import DEPARTMENTS
from ..extensions import mongo
from ..vectorstore.operations import VectorStoreOps
from ..utils.file_utils import save_file, delete_file, validate_file, get_file_extension

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = "pdf,docx,md,txt,xlsx,csv"


class DuplicateDocumentError(Exception):
    """Raise khi upload 1 file có nội dung TRÙNG (cùng SHA-256) với document đã
    tồn tại và chưa fail (Upload dedup RAG-011). Mang theo doc_id/tên file gốc
    để API trả 409 kèm thông tin document đã có, không ingest lại."""
    def __init__(self, existing_doc_id: str, existing_name: str):
        self.existing_doc_id = existing_doc_id
        self.existing_name = existing_name
        super().__init__(
            f"File trùng nội dung với document đã tồn tại (doc_id={existing_doc_id}, tên: {existing_name})."
        )


def _compute_file_hash(file) -> str:
    """SHA-256 nội dung file từ werkzeug FileStorage. Đọc theo block để không nạp
    cả file lớn vào RAM; seek về 0 trước VÀ sau để không phá luồng save_file()."""
    file.stream.seek(0)
    h = hashlib.sha256()
    for block in iter(lambda: file.stream.read(8192), b""):
        h.update(block)
    file.stream.seek(0)
    return h.hexdigest()


class DocumentService:
    """
    Quản lý vòng đời Document: upload → lưu metadata → trigger ingestion → delete.
    Hoạt động với MongoDB qua PyMongo (thông qua Flask-PyMongo extension).
    """

    def __init__(self, vector_ops: VectorStoreOps, upload_folder: str, bm25_index=None):
        self.vector_ops    = vector_ops
        self.upload_folder = upload_folder
        self.bm25_index    = bm25_index

    # ──────────────────────────────────────────────────────────
    # WRITE
    # ──────────────────────────────────────────────────────────

    def save_uploaded_file(self, file, user_id: str) -> dict:
        """
        Validate → dedup check → lưu file vật lý → tạo MongoDB record.
        Trả về dict chứa doc_id và file_path để caller trigger ingestion.
        Raise DuplicateDocumentError nếu nội dung file trùng document đã có.
        """
        # 1. Validate
        ok, err = validate_file(file, ALLOWED_EXTENSIONS)
        if not ok:
            raise ValueError(err)

        file_hash = _compute_file_hash(file)
        existing = mongo.db[Document.COLLECTION].find_one({
            "file_hash": file_hash,
            "status": {"$ne": Document.STATUS_FAILED},
        })
        if existing:
            raise DuplicateDocumentError(str(existing["_id"]), existing.get("original_name", ""))

        # 3. Lưu file vật lý (UUID filename)
        file_info = save_file(file, self.upload_folder)

        # 4. Tạo Document model
        doc = Document(
            filename      = file_info["uuid_filename"],
            original_name = file_info["original_name"],
            file_type     = file_info["extension"],
            file_size_kb  = file_info["size_kb"],
            uploaded_by   = user_id,
            file_hash     = file_hash,
        )

        # 5. Lưu vào MongoDB
        result = mongo.db[Document.COLLECTION].insert_one(doc.to_dict())
        doc_id = str(result.inserted_id)

        logger.info(f"Document saved: {doc_id} — {file_info['original_name']}")

        return {
            "doc_id":    doc_id,
            "filename":  file_info["original_name"],
            "file_path": file_info["file_path"],
            "file_type": file_info["extension"],
        }

    def update_status(self, doc_id: str, status: str,
                      error: str = None, chunk_count: int = None,
                      page_count: int = None, chroma_ids: list = None,
                      ocr_stats: dict = None):
        """Cập nhật trạng thái ingestion — được gọi bởi IngestionPipeline.

        ocr_stats: {"pages_total", "pages_needing_ocr", "pages_ocr_attempted",
        "pages_ocr_failed"} từ PDFParser.ocr_stats (chỉ PDF, None cho loại khác)
        — Dashboard OCR fail rate, """
        from bson import ObjectId
        update = {
            "status":     status,
            "updated_at": datetime.now(timezone.utc),
        }
        if error is not None:
            update["error_message"] = error
        if chunk_count is not None:
            update["chunk_count"] = chunk_count
        if page_count is not None:
            update["page_count"] = page_count
        if chroma_ids is not None:
            update["chroma_ids"] = chroma_ids
        if ocr_stats is not None:
            update["ocr_stats"] = ocr_stats

        mongo.db[Document.COLLECTION].update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": update},
        )
        logger.info(f"[{doc_id}] Status → {status}")

    def update_document_metadata(self, doc_id: str, document_status: str = None, document_type: str = None, department: str = None) -> bool:
        from bson import ObjectId

        if document_status is not None and document_status not in Document.VALID_DOC_STATUSES:
            raise ValueError(f"document_status không hợp lệ: {document_status!r}. Hợp lệ: {Document.VALID_DOC_STATUSES}")
        if document_type is not None and document_type not in Document.VALID_DOC_TYPES:
            raise ValueError(f"document_type không hợp lệ: {document_type!r}. Hợp lệ: {Document.VALID_DOC_TYPES}")
        if department is not None and department != "" and department not in DEPARTMENTS:
            raise ValueError(f"department không hợp lệ: {department!r}. Hợp lệ: {DEPARTMENTS} hoặc \"\" (bỏ gán).")

        update: dict = {"updated_at": datetime.now(timezone.utc)}
        chunk_updates: dict = {}
        if document_status is not None:
            update["document_status"] = document_status
            chunk_updates["document_status"] = document_status
        if document_type is not None:
            update["document_type"] = document_type
            chunk_updates["document_type"] = document_type
        if department is not None:
            update["department"] = department
            chunk_updates["department"] = department

        if not chunk_updates:
            return False

        result = mongo.db[Document.COLLECTION].update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": update},
        )
        if result.matched_count == 0:
            return False

        self.vector_ops.update_chunks_metadata(doc_id, chunk_updates)

        if self.bm25_index is not None:
            self.bm25_index.refresh(self.vector_ops)
            from ..core.retrieval import mark_bm25_dirty
            mark_bm25_dirty()
            logger.info(f"[{doc_id}] Đã refresh BM25 sau đổi metadata (đồng bộ ACL/filter).")

        logger.info(f"[{doc_id}] Đã gán metadata: {chunk_updates}")
        return True

    def delete_document(self, doc_id: str) -> bool:
        """Xóa file vật lý + chunks ChromaDB + MongoDB record."""
        from bson import ObjectId
        doc_data = mongo.db[Document.COLLECTION].find_one({"_id": ObjectId(doc_id)})
        if not doc_data:
            return False

        doc = Document.from_dict(doc_data)

        # 1. Xóa file vật lý
        file_path = os.path.join(self.upload_folder, doc.filename)
        delete_file(file_path)

        # 2. Xóa chunks trong ChromaDB
        deleted_count = self.vector_ops.delete_by_doc_id(doc_id)
        logger.info(f"[{doc_id}] Xóa {deleted_count} chunks khỏi ChromaDB.")

        if self.bm25_index is not None:
            self.bm25_index.refresh(self.vector_ops)
            # Báo các worker khác (multi-worker) tự refresh — fail-open nếu không
            # có Redis (xem core/bm25_sync.py). Worker này đã refresh cục bộ ở trên.
            from ..core.retrieval import mark_bm25_dirty
            mark_bm25_dirty()
            logger.info(f"[{doc_id}] Đã refresh BM25 index sau khi xóa.")

        # 3. Xóa MongoDB record
        mongo.db[Document.COLLECTION].delete_one({"_id": ObjectId(doc_id)})
        logger.info(f"[{doc_id}] Document đã xóa hoàn toàn.")
        return True

    # READ
    def get_documents(self, page: int = 1, limit: int = 20,
                      search: str = "", status: str = None,
                      acl_department: str = None, acl_bypass: bool = False) -> dict:
        """Danh sách documents với pagination, tìm kiếm, filter và Document-level ACL.

        acl_bypass=True (admin): bỏ qua ACL, thấy mọi document. Ngược lại chỉ
        thấy document department rỗng/chưa có field (44 doc cũ, fail-open theo
        quyết định đã chốt) HOẶC khớp đúng acl_department của user."""
        query: dict = {}
        if search:
            query["original_name"] = {"$regex": search, "$options": "i"}
        if status:
            query["status"] = status
        if not acl_bypass:
            dept = acl_department or ""
            dept_values = ["", None] if not dept else ["", None, dept]
            query["department"] = {"$in": dept_values}

        total  = mongo.db[Document.COLLECTION].count_documents(query)
        skip   = (page - 1) * limit
        cursor = (
            mongo.db[Document.COLLECTION]
            .find(query, {"chroma_ids": 0})   # Ẩn chroma_ids để response nhẹ hơn
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )

        docs = []
        for d in cursor:
            d["_id"] = str(d["_id"])
            docs.append(d)

        return {
            "documents": docs,
            "total":     total,
            "page":      page,
            "limit":     limit,
            "pages":     (total + limit - 1) // limit,
        }

    def _acl_query(self, acl_department: str, acl_bypass: bool) -> dict:
        """Điều kiện Mongo lọc theo Document-level ACL phòng ban (tái dùng cho
        list + search). admin (bypass) → không lọc. Nếu không: chỉ document phòng
        ban rỗng/chưa gán (fail-open, tài liệu cũ) HOẶC khớp đúng phòng ban user."""
        if acl_bypass:
            return {}
        dept = acl_department or ""
        return {"department": {"$in": ["", None] if not dept else ["", None, dept]}}

    def search_files(self, query: str, semantic_doc_ids: list[str] = None,
                     acl_department: str = None, acl_bypass: bool = False,
                     limit: int = 10) -> list[dict]:
        """
        Tìm FILE (không phải chunk) để dẫn hướng + tải về — dùng cho tính năng
        "tìm kiếm file". Kết hợp 2 tín hiệu, đều LỌC ACL phòng ban:
          1. Khớp TÊN FILE (regex, không phân biệt hoa thường) — ưu tiên hiển thị trước.
          2. Khớp NỘI DUNG: doc_id lấy từ semantic search (retriever, do route tính sẵn
             và đã lọc ACL ở tầng retrieve) — bắt được cả file mà tên không chứa từ khóa.
        Chỉ trả tài liệu status="done" (đã ingest xong, có file để tải). Mỗi kết quả
        kèm download_url + match_reason để UI hiển thị.
        """
        from bson import ObjectId

        acl = self._acl_query(acl_department, acl_bypass)
        matched: dict[str, tuple] = {}   # doc_id -> (doc_dict, match_reason)

        # 1. Khớp tên file
        name_query = {**acl, "status": "done",
                      "original_name": {"$regex": re.escape(query), "$options": "i"}}
        for d in mongo.db[Document.COLLECTION].find(name_query).limit(limit):
            matched[str(d["_id"])] = (d, "filename")

        # 2. Khớp nội dung (giữ đúng thứ tự relevance của semantic search)
        for doc_id in (semantic_doc_ids or []):
            if doc_id in matched:
                continue
            try:
                oid = ObjectId(doc_id)
            except Exception:
                continue
            d = mongo.db[Document.COLLECTION].find_one({"_id": oid, "status": "done", **acl})
            if d:
                matched[doc_id] = (d, "content")

        results = []
        for doc_id, (d, reason) in matched.items():
            results.append({
                "doc_id":        doc_id,
                "original_name": d.get("original_name"),
                "file_type":     d.get("file_type"),
                "document_type": d.get("document_type", ""),
                "department":    d.get("department", ""),
                "match_reason":  reason,
                "download_url":  f"/api/documents/{doc_id}/download",
            })
        return results[:limit]

    def get_document_by_id(self, doc_id: str, acl_department: str = None, acl_bypass: bool = True) -> dict | None:
        """Lấy chi tiết 1 document."""
        from bson import ObjectId
        try:
            doc_data = mongo.db[Document.COLLECTION].find_one({"_id": ObjectId(doc_id)})
        except Exception:
            return None
        if not doc_data:
            return None
        if not acl_bypass:
            doc_dept = doc_data.get("department") or ""
            user_dept = acl_department or ""
            if doc_dept and doc_dept != user_dept:
                # Coi như "không tìm thấy" — KHÔNG xác nhận tài liệu tồn tại
                # (khác 403 ở delete/update, đây là đọc/chống rò rỉ, xem
                # documents.py::get_document_status()).
                return None
        doc_data["_id"] = str(doc_data["_id"])
        return doc_data
