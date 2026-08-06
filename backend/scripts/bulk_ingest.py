"""
CLI script: ingest hàng loạt tài liệu từ 1 thư mục cục bộ vào RAG pipeline.

Dùng IngestionPipeline.ingest_batch() (`app/ingestion/pipeline.py`) — chỉ refresh
BM25 index MỘT LẦN sau cả batch, thay vì mỗi file như đường upload API đơn lẻ
(xem roadmap_tasklist/rag_core_quality_roadmap.md mục 6b để biết lý do).

Mỗi file được đăng ký qua DocumentService.save_uploaded_file() giống hệt luồng
upload API thật (tạo record MongoDB, copy file vào UPLOAD_FOLDER với tên UUID)
— tài liệu ingest qua script này sẽ hiện trong danh sách document như upload
bình thường qua UI.

Yêu cầu: MongoDB đang chạy (mongodb://localhost:27017 mặc định), Ollama KHÔNG
cần thiết cho ingestion (chỉ cần cho generation/QueryProcessor SLM lúc query).

Chạy:
    python backend/scripts/bulk_ingest.py <thư_mục_chứa_tài_liệu>
    python backend/scripts/bulk_ingest.py <thư_mục> --user-id my_user
"""
import sys
import os
import argparse
import time

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from werkzeug.datastructures import FileStorage

from app import create_app
from app.services.document_service import DocumentService
from app.ingestion.embedder.local_embedder import LocalEmbedder
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.ocr.tesseract_ocr_engine import TesseractOCREngine
from app.vectorstore.operations import VectorStoreOps
from app.core.retrieval import get_bm25_index

SUPPORTED_EXT = {".pdf", ".docx", ".md", ".txt"}


def collect_files(folder: str) -> list[str]:
    paths = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXT:
                paths.append(os.path.join(root, fn))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest tài liệu vào RAG pipeline.")
    parser.add_argument("folder", help="Thư mục chứa tài liệu cần ingest (quét đệ quy).")
    parser.add_argument("--user-id", default="bulk_ingest_cli", help="user_id gán cho các document.")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"❌ Thư mục không tồn tại: {args.folder}")
        sys.exit(1)

    file_paths = collect_files(args.folder)
    if not file_paths:
        print(f"❌ Không tìm thấy file nào ({', '.join(sorted(SUPPORTED_EXT))}) trong {args.folder}")
        sys.exit(1)

    print(f"📂 Tìm thấy {len(file_paths)} file trong {args.folder}")
    for p in file_paths:
        print(f"   - {os.path.relpath(p, args.folder)}")

    app = create_app("development")
    with app.app_context():
        vector_ops = VectorStoreOps()
        doc_service = DocumentService(vector_ops, upload_folder=app.config["UPLOAD_FOLDER"])
        print(f"\n🔧 Đang load LocalEmbedder ({app.config.get('LOCAL_EMBEDDING_MODEL')})... "
              f"(lần đầu có thể tải model từ HuggingFace Hub, ~2.2GB)")
        embedder = LocalEmbedder(model_name=app.config.get("LOCAL_EMBEDDING_MODEL"))

        ocr_engine = None
        if app.config.get("OCR_ENABLED", True):
            ocr_engine = TesseractOCREngine(
                tesseract_cmd=app.config.get("TESSERACT_CMD"),
                languages=app.config.get("OCR_LANGUAGES", "vie+eng"),
                dpi=app.config.get("OCR_DPI", 200),
            )

        pipeline = IngestionPipeline(
            embedder=embedder,
            vector_ops=vector_ops,
            doc_service=doc_service,
            bm25_index=get_bm25_index(),
            ocr_engine=ocr_engine,
        )

        # ── Bước 1: đăng ký từng file (MongoDB record + copy vào UPLOAD_FOLDER),
        #    giống hệt luồng upload API đơn lẻ ──────────────────────────────
        print(f"\n📝 Đang đăng ký {len(file_paths)} file vào MongoDB...")
        batch_files = []
        for path in file_paths:
            filename = os.path.basename(path)
            with open(path, "rb") as fh:
                storage = FileStorage(stream=fh, filename=filename)
                try:
                    doc_info = doc_service.save_uploaded_file(storage, user_id=args.user_id)
                except ValueError as e:
                    print(f"   ⚠️  Bỏ qua {filename}: {e}")
                    continue
            batch_files.append({
                "file_path": doc_info["file_path"],
                "doc_id":    doc_info["doc_id"],
                "filename":  doc_info["filename"],
            })
            print(f"   ✓ {doc_info['filename']} (doc_id={doc_info['doc_id']})")

        if not batch_files:
            print("❌ Không có file nào đăng ký thành công.")
            sys.exit(1)

        # ── Bước 2: ingest hàng loạt, refresh BM25 1 lần cuối ───────────────
        print(f"\n🚀 Bắt đầu ingest {len(batch_files)} file "
              f"(đồng bộ tuần tự, refresh BM25 1 lần sau cùng)...\n")
        t0 = time.time()
        results = pipeline.ingest_batch(batch_files)
        elapsed = time.time() - t0

        ok = sum(1 for r in results if r)
        fail = len(results) - ok

        print(f"\n{'=' * 60}")
        print(f"✅ Thành công: {ok}/{len(results)}  |  ❌ Thất bại: {fail}  |  ⏱️  {elapsed:.1f}s")
        if fail:
            print("\nFile thất bại (xem log phía trên hoặc trạng thái MongoDB để biết lỗi cụ thể):")
            for f, r in zip(batch_files, results):
                if not r:
                    print(f"   ❌ {f['filename']} (doc_id={f['doc_id']})")


if __name__ == "__main__":
    main()
