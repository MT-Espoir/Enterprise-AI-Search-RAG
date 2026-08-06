"""
Test BM25Index (tokenize/build/checksum-skip/search/filter doc_id) và HybridRetriever
(RRF fusion dedup + rank order) — dùng fake Retriever/BM25Index để cô lập, không cần
ChromaDB/embedder thật.

Chạy: python backend/tests/test_bm25_hybrid.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.retrieval import BM25Index, simple_tokenize
from app.core.retrieval import HybridRetriever

# ══════════════════════════════════════════════════════════════
# TEST 1: simple_tokenize — giữ dấu tiếng Việt, lowercase, tách âm tiết
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: simple_tokenize")

tokens = simple_tokenize("Mức Bồi Thường status_code 404!")
assert tokens == ["mức", "bồi", "thường", "status_code", "404"], f"❌ Tokenize sai: {tokens}"
print(f"  tokens = {tokens}")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 2: BM25Index — build, search tìm đúng chunk theo keyword
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: BM25Index.build() + search()")

chunks = [
    {"id": "c1", "text": "Lỗi status_code 404 xảy ra khi thiếu token xác thực.",
     "metadata": {"doc_id": "d1", "filename": "a.pdf", "page_num": 1}},
    {"id": "c2", "text": "Hướng dẫn cài đặt phần mềm trên Windows.",
     "metadata": {"doc_id": "d2", "filename": "b.pdf", "page_num": 1}},
    {"id": "c3", "text": "SQLite dùng để lưu dữ liệu cục bộ, không cần server riêng.",
     "metadata": {"doc_id": "d1", "filename": "a.pdf", "page_num": 2}},
]

index = BM25Index()
rebuilt = index.build(chunks)
assert rebuilt is True, "❌ Lần build đầu tiên phải trả True"

results = index.search("status_code 404", top_k=3)
assert results, "❌ search() không trả kết quả nào"
assert results[0]["id"] == "c1", f"❌ Top-1 phải là c1 (chứa status_code 404), nhận: {results[0]['id']}"
print(f"  Top-1 cho 'status_code 404': {results[0]['id']} (score={results[0]['score']:.3f})")

# Filter theo doc_id: chỉ chunk thuộc d1 (c1, c3)
filtered = index.search("SQLite", top_k=3, doc_id="d1")
assert all(r["metadata"]["doc_id"] == "d1" for r in filtered), "❌ Filter doc_id không hoạt động"
assert any(r["id"] == "c3" for r in filtered), "❌ Thiếu c3 trong kết quả filter theo d1"
print(f"  Filter doc_id='d1' → {[r['id'] for r in filtered]}")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 3: BM25Index — checksum guard, tránh rebuild thừa
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: checksum-skip khi corpus không đổi")

rebuilt_again = index.build(chunks)
assert rebuilt_again is False, "❌ Build lần 2 với corpus giống hệt phải trả False (skip)"
print("  Build lần 2 (corpus không đổi) → rebuilt=False ✅")

changed_chunks = chunks + [{"id": "c4", "text": "Chunk mới.", "metadata": {"doc_id": "d3"}}]
rebuilt_changed = index.build(changed_chunks)
assert rebuilt_changed is True, "❌ Build với corpus đã đổi phải trả True"
print("  Build khi corpus đổi (thêm c4) → rebuilt=True ✅")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 4: HybridRetriever — RRF fusion dedup + rank order
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: HybridRetriever RRF fusion")


class FakeEmbedder:
    def embed_query(self, question):
        return [0.0]


class FakeVectorOps:
    """Giả lập VectorStoreOps.query() — không cần ChromaDB thật."""
    def query(self, query_embedding, n_results=20, filter_metadata=None):
        # Vector search: chunk_A hạng 1 (cosine cao), chunk_B hạng 2
        return [
            {"id": "chunk_A", "text": "Nội dung A", "metadata": {"doc_id": "d1", "filename": "a.pdf"}, "score": 0.9},
            {"id": "chunk_B", "text": "Nội dung B", "metadata": {"doc_id": "d1", "filename": "a.pdf"}, "score": 0.5},
        ]


class FakeBM25Index:
    """Giả lập BM25Index.search() với kết quả cố định — cô lập test fusion logic
    khỏi hành vi tokenize/BM25Okapi thật."""
    # Chữ ký phải khớp BM25Index.search() thật — HybridRetriever truyền cả
    # acl_department/acl_bypass (thêm khi làm Document-level ACL). Stub thiếu 2
    # tham số này khiến test fail TypeError (lỗi có sẵn, phát hiện khi refactor).
    def search(self, question, top_k=20, doc_id=None, filters=None,
               acl_department=None, acl_bypass=False):
        # BM25: chunk_B hạng 1 (trùng với vector), chunk_C hạng 2 (BM25-only, vector bỏ sót)
        return [
            {"id": "chunk_B", "text": "Nội dung B", "metadata": {"doc_id": "d1", "filename": "a.pdf"}, "score": 5.0},
            {"id": "chunk_C", "text": "Nội dung C", "metadata": {"doc_id": "d1", "filename": "a.pdf"}, "score": 3.0},
        ]


retriever = HybridRetriever(
    ops=FakeVectorOps(), embedder=FakeEmbedder(), bm25_index=FakeBM25Index(),
    vector_pool_size=20, bm25_pool_size=20, rrf_k=60, top_k=10,
)

results = retriever.retrieve("câu hỏi bất kỳ")
ids = [c.chunk_id for c in results]

assert set(ids) == {"chunk_A", "chunk_B", "chunk_C"}, f"❌ Thiếu/thừa chunk sau fusion: {ids}"
print(f"  Chunks sau fusion: {ids}")

# chunk_B xuất hiện ở cả 2 list (rank 2 vector + rank 1 bm25) → RRF cao nhất → phải đứng đầu
assert ids[0] == "chunk_B", f"❌ chunk_B (trùng 2 nguồn) phải đứng đầu, nhận: {ids[0]}"
print("  chunk_B (overlap 2 nguồn) đứng đầu: ✅")

# chunk_A chỉ có trong vector -> vẫn phải còn trong kết quả (không bị BM25 loại bỏ)
chunk_a = next(c for c in results if c.chunk_id == "chunk_A")
assert chunk_a.metadata.get("vector_rank") == 1, "❌ chunk_A phải có vector_rank=1"
assert "bm25_rank" not in chunk_a.metadata, "❌ chunk_A không có trong BM25, không nên có bm25_rank"
print("  chunk_A (vector-only) giữ nguyên vector_rank=1: ✅")

# chunk_C chỉ có trong BM25 -> đúng case "BM25 bù cho vector" -> vẫn phải có trong kết quả
chunk_c = next(c for c in results if c.chunk_id == "chunk_C")
assert chunk_c.metadata.get("bm25_rank") == 2, "❌ chunk_C phải có bm25_rank=2"
assert "vector_rank" not in chunk_c.metadata, "❌ chunk_C không có trong vector, không nên có vector_rank"
print("  chunk_C (BM25-only, vector bỏ sót) vẫn có trong kết quả: ✅")

# Score phải sort giảm dần
scores = [c.score for c in results]
assert scores == sorted(scores, reverse=True), f"❌ Kết quả không sort giảm dần theo score: {scores}"
print(f"  Scores (RRF) giảm dần: {[round(s, 4) for s in scores]} ✅")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 5: BM25Index.search() — filter đa điều kiện (Phase 4 mục 4)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 5: BM25Index.search() — filters đa điều kiện")

meta_chunks = [
    {"id": "n1", "text": "Nghị định A còn hiệu lực", "metadata": {"doc_id": "d1", "document_status": "hieu_luc"}},
    {"id": "n2", "text": "Nghị định A bản cũ hết hiệu lực", "metadata": {"doc_id": "d1", "document_status": "het_hieu_luc"}},
    {"id": "n3", "text": "Nghị định B còn hiệu lực", "metadata": {"doc_id": "d2", "document_status": "hieu_luc"}},
]
meta_index = BM25Index()
meta_index.build(meta_chunks)

# filters đơn (không kèm doc_id) — chỉ document_status
only_status = meta_index.search("Nghị định", top_k=5, filters={"document_status": "hieu_luc"})
assert {r["id"] for r in only_status} == {"n1", "n3"}, f"❌ {only_status}"
print(f"  filters={{document_status: hieu_luc}} → {[r['id'] for r in only_status]} ✅")

# doc_id + filters cùng lúc (AND) — phải merge đúng
both = meta_index.search("Nghị định", top_k=5, doc_id="d1", filters={"document_status": "hieu_luc"})
assert [r["id"] for r in both] == ["n1"], f"❌ doc_id + filters phải AND lại còn đúng n1: {both}"
print(f"  doc_id='d1' + filters={{document_status: hieu_luc}} → {[r['id'] for r in both]} ✅")
print("  ✅ PASS\n")

print("=" * 60)
print("🎉 TẤT CẢ TESTS PASS!")
