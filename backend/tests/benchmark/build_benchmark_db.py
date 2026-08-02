"""
build_benchmark_db.py — Ingest TOÀN BỘ tài liệu test vào một ChromaDB riêng cho benchmark.

Khác với test_e2e_pipeline.py (chỉ embed 5 chunk đầu để né rate limit), script này
ingest đầy đủ document để bộ eval (evaluation_dataset.json) có thể query trên toàn bộ
nội dung thật. Chạy 1 lần trước khi chạy run_full_benchmark.py, hoặc chạy lại khi
evaluation_dataset.json/tài liệu nguồn thay đổi.

Dùng LocalEmbedder (BAAI/bge-m3, chạy CPU local qua sentence-transformers) thay vì
GoogleEmbedder — free tier Gemini không đủ quota để embed nhiều tài liệu (đặc biệt khi
TEST_FILES đã có tới 4 file, gồm 3 PDF). BGE-M3 không cần API key, không rate limit,
hỗ trợ tiếng Việt tốt.

⚠️  Nếu benchmark_chromadb/ đã có dữ liệu embed bằng GoogleEmbedder từ trước (số chiều
vector khác BGE-M3), XÓA thư mục benchmark_chromadb/ trước khi chạy lại — Chroma sẽ báo
lỗi dimension mismatch nếu trộn 2 loại embedding trong cùng 1 collection.

Chạy: python backend/tests/benchmark/build_benchmark_db.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.chunker.recursive_chunker import RecursiveChunker
from app.ingestion.embedder.local_embedder import LocalEmbedder
from app.vectorstore.operations import VectorStoreOps

DB_PATH = os.path.join(os.path.dirname(__file__), "benchmark_chromadb")
COLLECTION_NAME = "benchmark"

TEST_FILES = [
    {"filename": "System_Design_Deep_Feature_Analysis.md", "doc_id": "benchmark_doc_001"},
    {"filename": "1811.12808v3.pdf", "doc_id": "benchmark_doc_002"},
    {"filename": "ktmt_OnTap.pdf", "doc_id": "benchmark_doc_003"},
    {"filename": "THMLHK_223.pdf", "doc_id": "benchmark_doc_004"},
]


class MockVectorStoreOps(VectorStoreOps):
    """Override __init__ để inject collection trực tiếp, không cần Flask app context."""

    def __init__(self, collection):
        self.collection = collection


def build():
    print("🚀 Building benchmark ChromaDB (full document, không cắt bớt chunk)...")
    print("   Embedder: LocalEmbedder (BAAI/bge-m3, CPU) — lần đầu sẽ tải model (~2.2GB) từ HuggingFace Hub.")

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    vector_ops = MockVectorStoreOps(collection)

    md_parser = MarkdownParser()
    pdf_parser = PDFParser()
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)
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
        
        if filename.endswith(".md"):
            pages = md_parser.parse(test_file)
        elif filename.endswith(".pdf"):
            pages = pdf_parser.parse(test_file)
        else:
            print(f"   ⚠️ Unsupported file type: {filename}")
            continue
            
        print(f"   📄 Parsed: {len(pages)} page(s)")

        raw_chunks = chunker.chunk_pages(pages)
        total_raw_chunks += len(raw_chunks)
        print(f"   ✂️  Chunks: {len(raw_chunks)} (embed toàn bộ trên CPU — có thể mất vài phút với PDF dài, nhưng không giới hạn quota)")

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
    print(f"✅ Benchmark DB sẵn sàng tại {DB_PATH} (collection={COLLECTION_NAME})")


if __name__ == "__main__":
    build()
