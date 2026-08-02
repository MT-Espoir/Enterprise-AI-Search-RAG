"""
build_benchmark_db_legal.py — Ingest 10 văn bản luật/nghị định/biểu mẫu (đã xác
nhận KHÔNG bị lỗi OCR mất dấu, xem rag_core_quality_roadmap.md mục 6c/6d) vào
một ChromaDB benchmark RIÊNG (benchmark_chromadb_legal/), TÁCH BIỆT hoàn toàn
khỏi benchmark_chromadb/ (bộ 4 tài liệu IoT tiếng Anh gốc) — không đụng tới bộ
benchmark cũ đang dùng làm baseline hồi quy Phase 2.

Dùng chung config chunking/embedder với production (chunk_size=1000,
overlap=200, LocalEmbedder BAAI/bge-m3) — khác với build_benchmark_db.py cũ
(chunk_size=500/100) để phản ánh đúng hành vi production thật cho corpus mới.

Chạy: python backend/tests/benchmark/build_benchmark_db_legal.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.chunker.recursive_chunker import RecursiveChunker
from app.ingestion.embedder.local_embedder import LocalEmbedder
from app.vectorstore.operations import VectorStoreOps

DB_PATH = os.path.join(os.path.dirname(__file__), "benchmark_chromadb_legal")
COLLECTION_NAME = "benchmark_legal"

# Chỉ 10 văn bản đã xác nhận sạch 100% (is_scanned=False mọi trang) — xem
# rag_core_quality_roadmap.md mục 6c để biết vì sao KHÔNG dùng 12+11 file
# đã bị OCR mất dấu trong lần ingest sản xuất trước đó.
TEST_FILES = [
    {"filename": "24-2018-qh14.pdf", "doc_id": "legal_doc_001"},
    {"filename": "41-2024-qh15.pdf", "doc_id": "legal_doc_002"},
    {"filename": "luat-so-125.pdf", "doc_id": "legal_doc_003"},
    {"filename": "luat20-2023-qh15__e7e78.pdf", "doc_id": "legal_doc_004"},
    {"filename": "1452020nd-cp_25102023165145.docx", "doc_id": "legal_doc_005"},
    {"filename": "dieu-le-cong-ty-tnhh-1-thanh-vien-so-huu-ca-nhan .docx", "doc_id": "legal_doc_006"},
    {"filename": "dieu-le-cong-ty-tnhh-1-thanh-vien-so-huu-to-chuc.docx", "doc_id": "legal_doc_007"},
    {"filename": "mau-dieu-le-cong-ty-co-phan-mau-1.docx", "doc_id": "legal_doc_008"},
    {"filename": "mau-dieu-le-tnhh-2-thanh-vien-tro-len.docx", "doc_id": "legal_doc_009"},
    {"filename": "quyet-dinh-ban-hanh-quy-che-tien-luong_1005174201.docx", "doc_id": "legal_doc_010"},
]


class MockVectorStoreOps(VectorStoreOps):
    """Override __init__ để inject collection trực tiếp, không cần Flask app context."""

    def __init__(self, collection):
        self.collection = collection


def build():
    print("🚀 Building benchmark_legal ChromaDB (10 văn bản luật/nghị định/biểu mẫu sạch)...")

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    vector_ops = MockVectorStoreOps(collection)

    pdf_parser = PDFParser()  # KHÔNG truyền ocr_engine — cả 10 file đã xác nhận
    docx_parser = DocxParser()  # không cần OCR, nếu lỡ gặp trang scan sẽ báo lỗi rõ ràng thay vì OCR sai
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
    embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL"))

    total_raw_chunks = 0

    for file_info in TEST_FILES:
        filename = file_info["filename"]
        doc_id = file_info["doc_id"]
        test_file = os.path.join(os.path.dirname(__file__), "..", "Test_subject", filename)

        print(f"\nProcessing {filename}...")

        existing = collection.get(where={"doc_id": doc_id})
        if existing and existing["ids"]:
            print(f"   ⏩ Skipping {filename} (đã có {len(existing['ids'])} chunks trong DB, tránh embed lại)")
            total_raw_chunks += len(existing["ids"])
            continue

        if filename.endswith(".pdf"):
            pages = pdf_parser.parse(test_file)
        elif filename.endswith(".docx"):
            pages = docx_parser.parse(test_file)
        else:
            print(f"   ⚠️ Unsupported file type: {filename}")
            continue

        print(f"   📄 Parsed: {len(pages)} page(s)")

        raw_chunks = chunker.chunk_pages(pages)
        total_raw_chunks += len(raw_chunks)
        print(f"   ✂️  Chunks: {len(raw_chunks)}")

        if not raw_chunks:
            continue

        texts = [c["text"] for c in raw_chunks]
        embeddings = embedder.embed_batch(texts)

        chroma_chunks = []
        for i, (chunk, emb) in enumerate(zip(raw_chunks, embeddings)):
            meta = chunk["metadata"]
            page_num = meta.get("page_num", 1)
            chunk_index = meta.get("chunk_index", i)
            chroma_chunks.append(
                {
                    "id": f"{doc_id}_p{page_num}_c{chunk_index}",
                    "text": chunk["text"],
                    "embedding": emb,
                    "metadata": {
                        "doc_id": doc_id,
                        "filename": filename,
                        "page_num": page_num,
                        "chunk_index": chunk_index,
                    },
                }
            )

        stored = vector_ops.add_chunks(chroma_chunks)
        print(f"   💾 Stored {stored} chunks for {filename}")

    total = vector_ops.get_collection_count()
    print(f"\n   📊 DB total chunks: {total}")
    assert total == total_raw_chunks, "❌ Số chunk trong DB không khớp tổng chunk đã xử lý!"
    print(f"✅ Benchmark_legal DB sẵn sàng tại {DB_PATH} (collection={COLLECTION_NAME})")


if __name__ == "__main__":
    build()
