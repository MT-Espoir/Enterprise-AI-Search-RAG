"""
Test VectorStore (ChromaDB PersistentClient) - không cần Flask app context
Chạy: python backend/tests/test_vectorstore.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb

from app.vectorstore.operations import build_chroma_where

# ── Khởi tạo PersistentClient trực tiếp (không qua Flask) ────
DB_PATH = os.path.join(os.path.dirname(__file__), "chromadb_test_data")
client = chromadb.PersistentClient(path=DB_PATH)

COLLECTION_NAME = "test_collection"
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"✅ Collection '{COLLECTION_NAME}' sẵn sàng\n")

# ── Mock data: 4 chunks giả lập (dùng vector 4 chiều cho đơn giản) ───
mock_chunks = [
    {
        "id": "doc_001_chunk_0",
        "text": "Bên vi phạm hợp đồng phải bồi thường trong 30 ngày.",
        "embedding": [0.9, 0.1, 0.05, 0.02],
        "metadata": {"doc_id": "doc_001", "filename": "hop_dong.pdf", "page_num": 3}
    },
    {
        "id": "doc_001_chunk_1",
        "text": "Mức bồi thường tối thiểu là 10 triệu đồng theo điều 7.",
        "embedding": [0.85, 0.15, 0.08, 0.01],
        "metadata": {"doc_id": "doc_001", "filename": "hop_dong.pdf", "page_num": 5}
    },
    {
        "id": "doc_002_chunk_0",
        "text": "Thời tiết hôm nay rất đẹp và mát mẻ.",
        "embedding": [0.1, 0.9, 0.05, 0.02],
        "metadata": {"doc_id": "doc_002", "filename": "khac.pdf", "page_num": 1}
    },
    {
        "id": "doc_002_chunk_1",
        "text": "Dự báo thời tiết tuần tới có mưa nhẹ.",
        "embedding": [0.08, 0.88, 0.1, 0.03],
        "metadata": {"doc_id": "doc_002", "filename": "khac.pdf", "page_num": 2}
    },
]

# ── TEST 1: add_chunks ────────────────────────────────────────
print("=" * 60)
print("TEST 1: add_chunks()")

# Xóa sạch trước khi test để không bị duplicate
try:
    client.delete_collection(COLLECTION_NAME)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
except:
    pass

collection.add(
    ids        = [c["id"]        for c in mock_chunks],
    documents  = [c["text"]      for c in mock_chunks],
    embeddings = [c["embedding"] for c in mock_chunks],
    metadatas  = [c["metadata"]  for c in mock_chunks],
)
count = collection.count()
print(f"  Đã thêm: {len(mock_chunks)} chunks")
print(f"  Tổng trong DB: {count}")
assert count == 4, "❌ add_chunks thất bại!"
print("  ✅ PASS\n")

# ── TEST 2: query ─────────────────────────────────────────────
print("TEST 2: query() — tìm chunk liên quan đến hợp đồng")
query_vec = [0.88, 0.12, 0.06, 0.01]   # Vector gần với chunk hợp đồng

raw = collection.query(
    query_embeddings=[query_vec],
    n_results=2,
    include=["documents", "metadatas", "distances"]
)

ids       = raw["ids"][0]
docs      = raw["documents"][0]
distances = raw["distances"][0]
scores    = [round(1 - d, 4) for d in distances]

for i, (chunk_id, text, score) in enumerate(zip(ids, docs, scores)):
    print(f"  [Rank {i+1}] score={score}  id={chunk_id}")
    print(f"           text={text[:70]}...")

print(f"\n  Kỳ vọng: 2 chunks về hợp đồng có score cao nhất")
assert ids[0].startswith("doc_001"), "❌ Query trả về kết quả sai thứ tự!"
print("  ✅ PASS\n")

# ── TEST 3: query với filter_metadata ────────────────────────
print("TEST 3: query() với filter theo doc_id")
raw_filtered = collection.query(
    query_embeddings=[query_vec],
    n_results=2,
    where={"doc_id": "doc_002"},   # Chỉ tìm trong doc_002
    include=["documents", "distances"]
)
filtered_ids = raw_filtered["ids"][0]
print(f"  Filter doc_id=doc_002 → Tìm thấy {len(filtered_ids)} chunk(s)")
for fid in filtered_ids:
    print(f"    → {fid}")
assert all(i.startswith("doc_002") for i in filtered_ids), "❌ Filter sai!"
print("  ✅ PASS\n")

# ── TEST 4: delete_by_doc_id ──────────────────────────────────
print("TEST 4: delete_by_doc_id()")
before = collection.count()
collection.delete(where={"doc_id": "doc_001"})
after = collection.count()
deleted = before - after
print(f"  Trước: {before} chunks | Sau: {after} chunks | Đã xóa: {deleted}")
assert deleted == 2, f"❌ Phải xóa 2 chunks, nhưng xóa {deleted}"
print("  ✅ PASS\n")

# ── TEST 5: chunk_exists ──────────────────────────────────────
print("TEST 5: chunk_exists()")
exists     = len(collection.get(ids=["doc_002_chunk_0"])["ids"]) > 0
not_exists = len(collection.get(ids=["doc_001_chunk_0"])["ids"]) > 0   # Đã bị xóa
print(f"  doc_002_chunk_0 exists: {exists}    (kỳ vọng: True)")
print(f"  doc_001_chunk_0 exists: {not_exists}   (kỳ vọng: False)")
assert exists and not not_exists, "❌ chunk_exists sai!"
print("  ✅ PASS\n")

# ── TEST 6: build_chroma_where() — Phase 4 mục 4, metadata filtering đa chiều ──
print("TEST 6: build_chroma_where()")

assert build_chroma_where({}) is None, "❌ Dict rỗng phải trả None"
assert build_chroma_where({"doc_id": None}) is None, "❌ Toàn giá trị None phải trả None"
assert build_chroma_where({"doc_id": "d1"}) == {"doc_id": "d1"}, "❌ 1 điều kiện phải trả dict phẳng (tương thích ngược)"
where_multi = build_chroma_where({"doc_id": "d1", "document_status": "hieu_luc"})
assert where_multi == {"$and": [{"doc_id": "d1"}, {"document_status": "hieu_luc"}]}, f"❌ {where_multi}"
where_skip_none = build_chroma_where({"doc_id": "d1", "document_type": None})
assert where_skip_none == {"doc_id": "d1"}, f"❌ Phải loại bỏ key có giá trị None: {where_skip_none}"
print("  0/1/2+ điều kiện + loại bỏ None: ✅")
print("  ✅ PASS\n")

# ── TEST 7: multi-condition $and filter — xác nhận ChromaDB 1.5.9 THẬT hỗ trợ ──
print("TEST 7: query() với where=$and nhiều điều kiện (metadata đa chiều)")
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

multi_chunks = [
    {"id": "m1", "text": "Văn bản A còn hiệu lực", "embedding": [0.9, 0.1, 0.0, 0.0],
     "metadata": {"doc_id": "dA", "document_status": "hieu_luc", "document_type": "luat"}},
    {"id": "m2", "text": "Văn bản A đã hết hiệu lực (bản cũ)", "embedding": [0.88, 0.12, 0.0, 0.0],
     "metadata": {"doc_id": "dA", "document_status": "het_hieu_luc", "document_type": "luat"}},
    {"id": "m3", "text": "Văn bản B còn hiệu lực, khác loại", "embedding": [0.85, 0.1, 0.05, 0.0],
     "metadata": {"doc_id": "dB", "document_status": "hieu_luc", "document_type": "nghi_dinh"}},
]
collection.add(
    ids=[c["id"] for c in multi_chunks],
    documents=[c["text"] for c in multi_chunks],
    embeddings=[c["embedding"] for c in multi_chunks],
    metadatas=[c["metadata"] for c in multi_chunks],
)

where = build_chroma_where({"document_status": "hieu_luc", "document_type": "luat"})
raw = collection.query(query_embeddings=[[0.9, 0.1, 0.0, 0.0]], n_results=5, where=where, include=["metadatas"])
result_ids = raw["ids"][0]
assert result_ids == ["m1"], f"❌ $and 2 điều kiện phải chỉ khớp m1 (hiệu_lực + luật): {result_ids}"
print(f"  where={where} → {result_ids} (đúng chỉ m1) ✅")
print("  ✅ PASS\n")

# ── TEST 8: collection.update() metadata — cơ chế update_chunks_metadata() dựa vào ──
print("TEST 8: collection.update() merge metadata (nền tảng update_chunks_metadata)")
collection.update(ids=["m2"], metadatas=[{**multi_chunks[1]["metadata"], "document_status": "da_thay_the"}])
after = collection.get(ids=["m2"], include=["metadatas"])
assert after["metadatas"][0]["document_status"] == "da_thay_the", f"❌ Update không áp dụng: {after['metadatas'][0]}"
assert after["metadatas"][0]["document_type"] == "luat", "❌ Update phải GIỮ NGUYÊN field khác không đổi (merge thủ công phía Python)"
print(f"  Sau update: {after['metadatas'][0]} ✅")
print("  ✅ PASS\n")

# ── Dọn dẹp ──────────────────────────────────────────────────
client.delete_collection(COLLECTION_NAME)
print("=" * 60)
print("🎉 TẤT CẢ 8 TESTS ĐỀU PASS! VectorStore hoạt động đúng.")
