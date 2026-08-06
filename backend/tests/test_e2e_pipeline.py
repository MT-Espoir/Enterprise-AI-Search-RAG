"""
Test E2E Pipeline — IngestionPipeline + RAGPipeline
Dùng file Markdown thật từ test subject, ChromaDB PersistentClient (không cần server).

Chạy: python backend/tests/test_e2e_pipeline.py
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Thiếu GOOGLE_API_KEY trong .env")
    sys.exit(1)

import chromadb
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.chunker.recursive_chunker import RecursiveChunker
from app.ingestion.embedder.local_embedder import LocalEmbedder
from app.vectorstore.operations import VectorStoreOps
from app.core.retrieval import Retriever
from app.core.ranking import Reranker
from app.core.generation import LocalGenerator
from app.core.rag_pipeline import RAGPipeline

# ── Setup ChromaDB (PersistentClient — không cần server) ─────
DB_PATH = os.path.join(os.path.dirname(__file__), "e2e_chromadb")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
COLLECTION_NAME = "e2e_test"

# Reset collection để test sạch
try:
    chroma_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
chroma_collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# ── Mock VectorStoreOps (inject collection trực tiếp) ────────
class MockVectorStoreOps(VectorStoreOps):
    """Override __init__ để inject collection mà không cần Flask app."""
    def __init__(self, collection):
        self.collection = collection

vector_ops = MockVectorStoreOps(chroma_collection)

# ── Khởi tạo components ───────────────────────────────────────
print("🔧 Khởi tạo các components...")
import os
from app.ingestion.embedder.google_embedder import GoogleEmbedder

API_KEY = os.getenv("GOOGLE_API_KEY")
embedder  = GoogleEmbedder(api_key=API_KEY)
chunker   = RecursiveChunker(chunk_size=500, chunk_overlap=100)
parser    = MarkdownParser()
retriever = Retriever(ops=vector_ops, embedder=embedder, top_k=10)
reranker  = Reranker.get_instance(top_k=3)
generator = LocalGenerator()
rag       = RAGPipeline(retriever=retriever, reranker=reranker, generator=generator)
print("   ✅ Tất cả components sẵn sàng\n")

# ══════════════════════════════════════════════════════════════
# TEST 1: Ingestion — Parse → Chunk → Embed → Store
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Ingestion Pipeline (Parse → Chunk → Embed → Store)")

TEST_FILE = os.path.join(os.path.dirname(__file__), "Test_subject/System_Design_Deep_Feature_Analysis.md")
DOC_ID    = "test_doc_001"
FILENAME  = "System_Design_Deep_Feature_Analysis.md"

# Parse
pages = parser.parse(TEST_FILE)
print(f"  📄 Parsed: {len(pages)} page(s)")

# Chunk
raw_chunks = chunker.chunk_pages(pages)
print(f"  ✂️  Chunks: {len(raw_chunks)}")

# Để tránh rate limit API (15 request/phút) của Free Tier, ta chỉ test 5 chunk đầu tiên
raw_chunks = raw_chunks[:5]

# Embed + build ChromaDB records
print(f"  🔢 Đang embed {len(raw_chunks)} chunks (có thể mất vài giây)...")
texts      = [c["text"] for c in raw_chunks]
embeddings = embedder.embed_batch(texts)

chroma_chunks = []
for i, (chunk, emb) in enumerate(zip(raw_chunks, embeddings)):
    meta = chunk["metadata"]
    chroma_chunks.append({
        "id":        f"{DOC_ID}_p{meta.get('page_num',1)}_c{meta.get('chunk_index',i)}",
        "text":      chunk["text"],
        "embedding": emb,
        "metadata": {
            "doc_id":      DOC_ID,
            "filename":    FILENAME,
            "page_num":    meta.get("page_num", 1),
            "chunk_index": meta.get("chunk_index", i),
        },
    })

# Store
stored = vector_ops.add_chunks(chroma_chunks)
total_in_db = vector_ops.get_collection_count()
print(f"  💾 Stored: {stored} chunks | DB total: {total_in_db}")
assert total_in_db == len(raw_chunks), "❌ Số chunk trong DB không khớp!"
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 2: RAG Pipeline — Query → Rerank → Generate
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: RAG Pipeline (Retrieve → Rerank → Generate)")

question = "What is the Edge Sensor Telemetry Ingestion process?"
print(f"  ❓ Question: {question}\n")

result = rag.run(question)

print(f"  📊 Candidates retrieved: {result['candidates']}")
print(f"  📝 Answer ({result['tokens_used']} tokens):")
print(f"  {result['answer'][:400]}...")
print(f"\n  📚 Sources ({len(result['sources'])}):")
for src in result["sources"]:
    print(f"    → {src['filename']} (page {src['page']})")

assert result["answer"], "❌ Answer rỗng!"
assert result["candidates"] > 0, "❌ Không tìm được candidates!"
print("\n  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 3: RAG Routing (Summary Intent)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: RAG Routing (Summary Intent)")

question_summary = "Hãy tóm tắt nội dung chính của tài liệu này"
print(f"  ❓ Question: {question_summary}\n")

# Bắt buộc phải truyền doc_id cho Summary intent
result_summary = rag.run(question_summary, doc_id=DOC_ID)

print(f"  📊 Candidates retrieved: {result_summary['candidates']} (bằng số chunk fetch thẳng, không rerank)")
print(f"  📝 Answer ({result_summary['tokens_used']} tokens):")
print(f"  {result_summary['answer'][:400]}...")
print(f"\n  📚 Sources ({len(result_summary['sources'])}):")
for src in result_summary["sources"]:
    print(f"    → {src['filename']} (page {src['page']})")

assert result_summary["answer"], "❌ Answer rỗng!"
assert result_summary["candidates"] > 0, "❌ Không tìm được chunks nào cho summary!"
print("\n  ✅ PASS\n")

# ── Dọn dẹp ──────────────────────────────────────────────────
# chroma_client.delete_collection(COLLECTION_NAME)
print("=" * 60)
print("🎉 E2E PIPELINE TEST HOÀN THÀNH — TẤT CẢ PASS!")
