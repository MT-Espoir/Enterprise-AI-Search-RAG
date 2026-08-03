import logging
import threading
import os
from pathlib import Path

from .parsers import PDFParser, DocxParser, MarkdownParser, ExcelParser, CsvParser
from .parsers.base_parser import BaseParser
from .chunker.recursive_chunker import RecursiveChunker
from .embedder.google_embedder import GoogleEmbedder
from ..vectorstore.operations import VectorStoreOps
from ..models.document import Document

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Điều phối toàn bộ pipeline xử lý tài liệu:
      Parse → Chunk → Embed → Store vào ChromaDB
    """

    PARSER_MAP: dict[str, type[BaseParser]] = {
        ".pdf":  PDFParser,
        ".docx": DocxParser,
        ".md":   MarkdownParser,
        ".txt":  MarkdownParser,   # txt dùng chung MarkdownParser
        ".xlsx": ExcelParser,
        ".csv":  CsvParser,
    }

    def __init__(self, vector_ops: VectorStoreOps, doc_service, embedder,
                 chunk_size: int = 1000, chunk_overlap: int = 200, bm25_index=None, ocr_engine=None):
        self.vector_ops  = vector_ops
        self.doc_service = doc_service
        self.embedder    = embedder
        self.bm25_index  = bm25_index
        self.ocr_engine  = ocr_engine
        self.chunker     = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # PUBLIC API
    def ingest(self, file_path: str, doc_id: str, filename: str, refresh_bm25: bool = True) -> bool:
        """
        Chạy pipeline đồng bộ (blocking).
        Trả về True nếu thành công, False nếu thất bại.
        Thường được gọi bên trong ingest_async().

        refresh_bm25: mặc định True (hành vi cũ, dùng cho upload đơn lẻ qua API).
        Đặt False khi gọi từ ingest_batch() — refresh BM25 rebuild TOÀN BỘ index
        (O(N) theo tổng số chunk hệ thống).
        """
        try:
            self.doc_service.update_status(doc_id, "processing")
            logger.info(f"[{doc_id}] Bắt đầu ingestion: {filename}")

            # ── Bước 1: Parse ──────────────────────────────
            parser = self._get_parser(filename)
            pages  = parser.parse(file_path)
            if not pages:
                raise ValueError("Parser không trích xuất được nội dung từ file.")
            logger.info(f"[{doc_id}] Parsed: {len(pages)} trang")

            # ── Bước 2: Chunk ──────────────────────────────
            raw_chunks = self.chunker.chunk_pages(pages)
            if not raw_chunks:
                raise ValueError("Chunker không tạo ra chunk nào.")
            logger.info(f"[{doc_id}] Chunks: {len(raw_chunks)}")

            # ── Bước 3: Embed + Build ChromaDB records ─────
            texts      = [c["text"] for c in raw_chunks]
            embeddings = self.embedder.embed_batch(texts)

            total = len(raw_chunks)
            chroma_chunks = []
            for i, (chunk, embedding) in enumerate(zip(raw_chunks, embeddings)):
                meta = chunk["metadata"]
                page_num    = meta.get("page_num", 1)
                chunk_index = meta.get("chunk_index", i)

                chroma_chunks.append({
                    "id":        f"{doc_id}_p{page_num}_c{chunk_index}",
                    "text":      chunk["text"],
                    "embedding": embedding,
                    "metadata": {
                        "doc_id":        doc_id,
                        "filename":      filename,
                        "page_num":      page_num,
                        "chunk_index":   chunk_index,
                        "total_chunks":  total,
                        "document_status": Document.DOC_STATUS_CHUA_XAC_DINH,
                        "document_type":   "",
                        "department":      "",
                        **{k: v for k, v in meta.items()
                           if k not in ("page_num", "chunk_index")},
                    },
                })

            # ── Bước 4: Lưu vào ChromaDB ───────────────────
            self.vector_ops.add_chunks(chroma_chunks)
            logger.info(f"[{doc_id}] Đã lưu {total} chunks vào ChromaDB.")

            if self.bm25_index is not None and refresh_bm25:
                self.bm25_index.refresh(self.vector_ops)
                # Báo các worker khác (multi-worker) tự refresh — fail-open nếu
                # không có Redis (xem core/bm25_sync.py). Worker này đã refresh cục bộ.
                from ..core.bm25_sync import mark_bm25_dirty
                mark_bm25_dirty()
                logger.info(f"[{doc_id}] Đã refresh BM25 index.")

            # ── Bước 5: Cập nhật trạng thái MongoDB ────────
            # ocr_stats chỉ có ở PDFParser (Dashboard OCR fail rate, mục 6m) —
            # None cho parser khác (docx/md/txt), getattr an toàn cho cả 2 trường hợp.
            ocr_stats = getattr(parser, "ocr_stats", None)
            self.doc_service.update_status(
                doc_id, "done",
                chunk_count=total,
                page_count=len(pages),
                chroma_ids=[c["id"] for c in chroma_chunks],
                ocr_stats=ocr_stats,
            )
            return True

        except Exception as exc:
            logger.error(f"[{doc_id}] Ingestion thất bại: {exc}", exc_info=True)
            self.doc_service.update_status(doc_id, "failed", error=str(exc))
            # Dọn file vật lý CHỈ khi ingest THẤT BẠI. Khi thành công phải GIỮ LẠI file
            # gốc (trong upload_folder, tên = uuid_filename) để phục vụ tính năng
            # tìm & tải file (GET /api/documents/<id>/download). delete_document() sẽ
            # dọn file này khi user chủ động xóa tài liệu.
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            return False

    def ingest_batch(self, files: list[dict]) -> list[bool]:
        """
        Ingest nhiều tài liệu LIÊN TIẾP (đồng bộ, không dùng threading), chỉ
        refresh BM25 index MỘT LẦN sau khi toàn bộ batch xong — thay vì mỗi
        tài liệu 

        files: list[{"file_path": str, "doc_id": str, "filename": str}]
        Trả về list[bool] kết quả từng file, đúng thứ tự với `files`.
        """
        results = [
            self.ingest(f["file_path"], f["doc_id"], f["filename"], refresh_bm25=False)
            for f in files
        ]

        if self.bm25_index is not None:
            self.bm25_index.refresh(self.vector_ops)
            from ..core.bm25_sync import mark_bm25_dirty
            mark_bm25_dirty()
            logger.info(f"Batch ingest xong {len(files)} tài liệu — đã refresh BM25 index 1 lần.")

        return results

    def ingest_async(self, file_path: str, doc_id: str, filename: str) -> threading.Thread:
        """
        Chạy pipeline trong background thread (non-blocking).
        API endpoint gọi hàm này để trả về HTTP 202 ngay lập tức.
        """
        thread = threading.Thread(
            target=self.ingest,
            args=(file_path, doc_id, filename),
            daemon=True,   # Tự kết thúc khi main process dừng
            name=f"ingest-{doc_id}",
        )
        thread.start()
        logger.info(f"[{doc_id}] Background ingestion thread đã khởi động.")
        return thread

    # PRIVATE
    def _get_parser(self, filename: str) -> BaseParser:
        """Chọn parser phù hợp dựa trên extension của filename."""
        ext = Path(filename).suffix.lower()
        parser_cls = self.PARSER_MAP.get(ext)
        if parser_cls is None:
            supported = ", ".join(self.PARSER_MAP.keys())
            raise ValueError(f"Định dạng '{ext}' không được hỗ trợ. Hỗ trợ: {supported}")
        if parser_cls is PDFParser:
            return PDFParser(ocr_engine=self.ocr_engine)
        return parser_cls()
