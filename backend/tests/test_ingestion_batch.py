"""
Test IngestionPipeline.ingest_batch() — xác nhận batch ingest chỉ refresh BM25
index ĐÚNG 1 LẦN cho cả batch (thay vì mỗi tài liệu), và ingest()/ingest_async()
đơn lẻ vẫn giữ hành vi cũ (refresh sau mỗi file).

Bối cảnh: BM25Index.refresh() rebuild TOÀN BỘ index (O(N) theo tổng số chunk
hệ thống) mỗi lần gọi. Gọi nó sau mỗi file khi ingest hàng loạt (vd hàng nghìn
tài liệu) sẽ thành O(N²) tích lũy — xem roadmap_tasklist/rag_core_quality_roadmap.md
mục 6b. ingest_batch() sửa lỗi này bằng cách chỉ refresh 1 lần sau cả batch.

Chạy: python backend/tests/test_ingestion_batch.py
"""
import sys, os, shutil, tempfile
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ingestion.pipeline import IngestionPipeline

TEST_FILE = os.path.join(os.path.dirname(__file__), "Test_subject/System_Design_Deep_Feature_Analysis.md")


def temp_copy_of_test_file() -> str:
    """ingest() xoá file ở finally sau khi xử lý — copy ra temp mỗi lần
    để không xoá mất fixture gốc dùng chung với test_e2e_pipeline.py."""
    fd, path = tempfile.mkstemp(suffix=".md")
    os.close(fd)
    shutil.copyfile(TEST_FILE, path)
    return path


class FakeEmbedder:
    def embed_batch(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]


class FakeVectorOps:
    def __init__(self):
        self.stored_chunks = []

    def add_chunks(self, chunks):
        self.stored_chunks.extend(chunks)
        return len(chunks)

    def get_all_chunks(self):
        return [{"id": c["id"], "text": c["text"], "metadata": c["metadata"]} for c in self.stored_chunks]


class FakeDocService:
    def __init__(self):
        self.statuses = {}

    def update_status(self, doc_id, status, **kwargs):
        self.statuses[doc_id] = status


class FakeBM25Index:
    def __init__(self):
        self.refresh_call_count = 0

    def refresh(self, vector_ops):
        self.refresh_call_count += 1


def make_pipeline():
    vector_ops = FakeVectorOps()
    doc_service = FakeDocService()
    bm25_index = FakeBM25Index()
    pipeline = IngestionPipeline(
        vector_ops=vector_ops,
        doc_service=doc_service,
        embedder=FakeEmbedder(),
        bm25_index=bm25_index,
    )
    return pipeline, vector_ops, doc_service, bm25_index


# ══════════════════════════════════════════════════════════════
# TEST 1: ingest() đơn lẻ — refresh BM25 sau MỖI file (hành vi cũ, không đổi)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: ingest() đơn lẻ vẫn refresh BM25 mỗi lần gọi")

pipeline, vector_ops, doc_service, bm25_index = make_pipeline()

for i in range(3):
    ok = pipeline.ingest(file_path=temp_copy_of_test_file(), doc_id=f"doc_{i}", filename="a.md")
    assert ok, f"❌ ingest() thất bại ở file {i}"

print(f"  refresh_call_count sau 3 lần ingest() riêng lẻ = {bm25_index.refresh_call_count}")
assert bm25_index.refresh_call_count == 3, f"❌ Kỳ vọng refresh 3 lần (1/file), nhận {bm25_index.refresh_call_count}"
print("  ✅ PASS — hành vi ingest() đơn lẻ giữ nguyên (1 refresh/file)\n")


# ══════════════════════════════════════════════════════════════
# TEST 2: ingest_batch() — chỉ refresh BM25 ĐÚNG 1 LẦN cho cả batch
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: ingest_batch() chỉ refresh BM25 1 lần cho cả batch")

pipeline2, vector_ops2, doc_service2, bm25_index2 = make_pipeline()

files = [
    {"file_path": temp_copy_of_test_file(), "doc_id": f"batch_doc_{i}", "filename": "a.md"}
    for i in range(5)
]
results = pipeline2.ingest_batch(files)

print(f"  Kết quả từng file: {results}")
print(f"  refresh_call_count sau ingest_batch(5 file) = {bm25_index2.refresh_call_count}")

assert len(results) == 5, f"❌ Kỳ vọng 5 kết quả, nhận {len(results)}"
assert all(results), f"❌ Có file ingest thất bại: {results}"
assert bm25_index2.refresh_call_count == 1, (
    f"❌ Kỳ vọng refresh BM25 ĐÚNG 1 lần cho cả batch, nhận {bm25_index2.refresh_call_count} "
    "(đây chính là bug O(N²) nếu quay lại refresh mỗi file)"
)
for doc_id in [f["doc_id"] for f in files]:
    assert doc_service2.statuses[doc_id] == "done", f"❌ {doc_id} không ở trạng thái done"

print("  ✅ PASS — ingest_batch() refresh BM25 đúng 1 lần, không phải O(N)\n")

print("=" * 60)
print("🎉 TEST INGESTION BATCH HOÀN THÀNH — TẤT CẢ PASS!")
