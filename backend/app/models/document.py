from datetime import datetime, timezone
from bson import ObjectId

from .department import DEPARTMENTS, DEFAULT_DEPARTMENT

class Document:
    COLLECTION = "documents"
    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    # KHÁC với `status` ở trên (trạng thái xử lý ingest). document_status là trạng
    # thái HIỆU LỰC PHÁP LÝ, do user tự gán thủ công qua API riêng — KHÔNG tự động
    # suy luận từ nội dung. Mặc định "chua_xac_dinh" cho MỌI document để nhất quán, không bắt buộc gán.
    DOC_STATUS_HIEU_LUC = "hieu_luc"
    DOC_STATUS_HET_HIEU_LUC = "het_hieu_luc"
    DOC_STATUS_DA_THAY_THE = "da_thay_the"
    DOC_STATUS_CHUA_XAC_DINH = "chua_xac_dinh"
    VALID_DOC_STATUSES = (DOC_STATUS_HIEU_LUC, DOC_STATUS_HET_HIEU_LUC, DOC_STATUS_DA_THAY_THE, DOC_STATUS_CHUA_XAC_DINH)

    DOC_TYPE_LUAT = "luat"
    DOC_TYPE_NGHI_DINH = "nghi_dinh"
    DOC_TYPE_BIEU_MAU = "bieu_mau"
    DOC_TYPE_QUY_CHE = "quy_che"
    DOC_TYPE_KHAC = "khac"
    VALID_DOC_TYPES = (DOC_TYPE_LUAT, DOC_TYPE_NGHI_DINH, DOC_TYPE_BIEU_MAU, DOC_TYPE_QUY_CHE, DOC_TYPE_KHAC)

    def __init__(self, filename, original_name, file_type, file_size_kb, uploaded_by, _id=None, status=STATUS_PENDING, error_message=None, chunk_count=0, page_count=None, chroma_ids=None, created_at=None, updated_at=None, document_status=DOC_STATUS_CHUA_XAC_DINH, document_type=None, department=None, ocr_stats=None, file_hash=None):
        self._id = _id or ObjectId()
        self.filename = filename
        self.original_name = original_name
        self.file_type = file_type
        self.file_size_kb = file_size_kb
        self.uploaded_by = str(uploaded_by)  # user_id
        self.status = status
        self.error_message = error_message
        self.chunk_count = chunk_count
        self.page_count = page_count
        self.chroma_ids = chroma_ids or []
        self.document_status = document_status or self.DOC_STATUS_CHUA_XAC_DINH
        self.document_type = document_type
        # Phòng ban được phép truy vấn tài liệu này (Document-level ACL) — ""
        # (mặc định) = chưa gán = mở cho mọi user, KHÔNG BAO GIỜ None (ChromaDB
        # metadata không nhận None). Admin gán thủ công qua PATCH .../metadata.
        self.department = department if department in DEPARTMENTS else DEFAULT_DEPARTMENT
        # Thống kê OCR — chỉ có ý nghĩa với PDF đã qua
        # OCR engine, None cho tài liệu khác (docx/md/txt hoặc PDF không cần OCR).
        self.ocr_stats = ocr_stats
        # SHA-256 nội dung file — chống ingest
        self.file_hash = file_hash

        now = datetime.now(timezone.utc)
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self) -> dict:
        return {
            "_id": self._id,
            "filename": self.filename,
            "original_name": self.original_name,
            "file_type": self.file_type,
            "file_size_kb": self.file_size_kb,
            "uploaded_by": self.uploaded_by,
            "status": self.status,
            "error_message": self.error_message,
            "chunk_count": self.chunk_count,
            "page_count": self.page_count,
            "chroma_ids": self.chroma_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "document_status": self.document_status,
            "document_type": self.document_type,
            "department": self.department,
            "ocr_stats": self.ocr_stats,
            "file_hash": self.file_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        if not data:
            return None
        return cls(
            _id=data.get("_id"),
            filename=data.get("filename"),
            original_name=data.get("original_name"),
            file_type=data.get("file_type"),
            file_size_kb=data.get("file_size_kb"),
            uploaded_by=data.get("uploaded_by"),
            status=data.get("status", cls.STATUS_PENDING),
            error_message=data.get("error_message"),
            chunk_count=data.get("chunk_count", 0),
            page_count=data.get("page_count"),
            chroma_ids=data.get("chroma_ids", []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            document_status=data.get("document_status", cls.DOC_STATUS_CHUA_XAC_DINH),
            document_type=data.get("document_type"),
            department=data.get("department", DEFAULT_DEPARTMENT),
            ocr_stats=data.get("ocr_stats"),
            file_hash=data.get("file_hash"),
        )
