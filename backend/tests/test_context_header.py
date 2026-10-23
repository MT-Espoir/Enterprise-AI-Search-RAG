"""
Test header ngữ cảnh tài liệu (app/core/context_header.py) ở đủ 5 bề mặt: hàm dùng
chung, ingest (embedding + metadata), BM25, Reranker, Generator — và model Document.
Trọng tâm: killswitch TẮT phải giữ NGUYÊN hành vi cũ ở mọi bề mặt; nội dung chunk lưu
vào ChromaDB / trả cho Generator luôn là bản gốc (header chỉ ở embedding/BM25/rerank
và nhãn nguồn). Không cần Ollama/ChromaDB/model thật.

Chạy: python backend/tests/test_context_header.py
"""
import sys, os, tempfile
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core import context_header as ctx
from app.core.retrieval.bm25_index import BM25Index
from app.core.ranking.reranker import Reranker
from app.core.generation.base_generator import BaseGenerator
from app.core.schemas import RetrievedChunk
from app.ingestion.pipeline import IngestionPipeline
from app.models.document import Document

HEADER = "[Báo cáo tài chính riêng năm 2016 của Tổng Công ty Vinaconex]"


# ══════════════════════════════════════════════════════════════
# TEST 1: Hàm dùng chung + killswitch
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: build_context_text / header_of / context_text_for")

assert ctx.build_context_text(HEADER, "nội dung") == f"{HEADER}\nnội dung"
assert ctx.build_context_text("", "nội dung") == "nội dung"
assert ctx.build_context_text(None, "nội dung") == "nội dung"
assert ctx.build_context_text("   ", "nội dung") == "nội dung"

meta = {"context_header": HEADER}
ctx.set_enabled(False)
assert ctx.header_of(meta) == "" and ctx.context_text_for("nội dung", meta) == "nội dung"
ctx.set_enabled(True)
assert ctx.header_of(meta) == HEADER and ctx.context_text_for("nội dung", meta) == f"{HEADER}\nnội dung"
assert ctx.header_of({}) == "" and ctx.header_of(None) == ""
ctx.set_enabled(False)
print("  ✅ PASS — killswitch tắt: query-time bỏ qua header; bật: ghép đúng 1 công thức\n")


# ══════════════════════════════════════════════════════════════
# TEST 2: Ingest — embedding có header, nội dung lưu là bản GỐC
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: IngestionPipeline ghép header vào embedding")


class RecordingEmbedder:
    def __init__(self):
        self.texts = []

    def embed_batch(self, texts):
        self.texts.extend(texts)
        return [[0.0, 0.0, 0.0] for _ in texts]


class FakeVectorOps:
    def __init__(self):
        self.stored = []

    def add_chunks(self, chunks):
        self.stored.extend(chunks)
        return len(chunks)


class FakeDocService:
    def update_status(self, doc_id, status, **kwargs):
        self.status = status


def ingest_once(enabled: bool, header: str | None):
    ctx.set_enabled(enabled)
    fd, path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("# Thuyết minh\n\nDoanh thu thuần năm nay đạt 1.234 tỷ đồng, tăng so với năm trước.\n")
    embedder, ops, docs = RecordingEmbedder(), FakeVectorOps(), FakeDocService()
    ok = IngestionPipeline(vector_ops=ops, doc_service=docs, embedder=embedder).ingest(
        path, "doc1", "bctc.md", context_header=header,
    )
    os.remove(path)
    assert ok and docs.status == "done"
    return embedder.texts, ops.stored


texts, stored = ingest_once(enabled=False, header=HEADER)
assert all(not t.startswith("[") for t in texts), "killswitch tắt KHÔNG được ghép header vào embedding"
assert all("context_header" not in c["metadata"] for c in stored)

texts, stored = ingest_once(enabled=True, header=HEADER)
assert all(t.startswith(HEADER + "\n") for t in texts), texts
assert all(c["metadata"]["context_header"] == HEADER for c in stored)
assert all(not c["text"].startswith("[") for c in stored), "nội dung lưu vào ChromaDB phải là bản GỐC"

texts, stored = ingest_once(enabled=True, header=None)
assert all(not t.startswith("[") for t in texts)
assert all("context_header" not in c["metadata"] for c in stored), "không có header thì không ghi metadata"
ctx.set_enabled(False)
print("  ✅ PASS — metadata.context_header chỉ có khi embedding thật sự có header\n")


# ══════════════════════════════════════════════════════════════
# TEST 3: BM25 — header được đánh chỉ mục, text trả về là bản gốc, checksum đổi
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: BM25Index đánh chỉ mục header")

chunks = [
    {"id": "a", "text": "Doanh thu thuần đạt 1.234 tỷ đồng", "metadata": {"doc_id": "d1", "context_header": HEADER}},
    {"id": "b", "text": "Chi phí quản lý doanh nghiệp tăng", "metadata": {"doc_id": "d2"}},
    {"id": "c", "text": "Lợi nhuận gộp về bán hàng", "metadata": {"doc_id": "d3"}},
]
bm25 = BM25Index()

ctx.set_enabled(False)
assert bm25.build(chunks) is True
assert [h["id"] for h in bm25.search("vinaconex 2016", top_k=3) if h["score"] > 0] == [], \
    "killswitch tắt: từ chỉ có trong header không được khớp"

ctx.set_enabled(True)
assert bm25.build(chunks) is True, "bật killswitch phải đổi checksum → rebuild"
hits = [h for h in bm25.search("vinaconex 2016", top_k=3) if h["score"] > 0]
assert [h["id"] for h in hits] == ["a"], hits
assert hits[0]["text"] == "Doanh thu thuần đạt 1.234 tỷ đồng", "text trả về phải là bản GỐC"
assert bm25.build(chunks) is False, "không đổi gì thì không rebuild"

changed = [dict(c, metadata=dict(c["metadata"])) for c in chunks]
changed[0]["metadata"]["context_header"] = "[Báo cáo tài chính riêng năm 2017 của Tổng Công ty Vinaconex]"
assert bm25.build(changed) is True, "đổi header phải đổi checksum → rebuild"
ctx.set_enabled(False)
print("  ✅ PASS — header khớp được khi bật, text gốc, checksum theo header + killswitch\n")


# ══════════════════════════════════════════════════════════════
# TEST 4: Reranker — cross-encoder nhận header + nội dung (không load model thật)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: Reranker nhận chuỗi có header")


class RecordingModel:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return [float(i) for i in range(len(pairs))]


def make_chunk(cid, text, header=None):
    meta = {"context_header": header} if header else {}
    return RetrievedChunk(chunk_id=cid, doc_id="d", filename="f.pdf", page=1, text=text, score=0.0, metadata=meta)


reranker = Reranker.__new__(Reranker)
reranker.model, reranker.top_k = RecordingModel(), 5

ctx.set_enabled(True)
out = reranker.rerank("doanh thu 2016", [make_chunk("a", "nội dung A", HEADER), make_chunk("b", "nội dung B")])
assert reranker.model.pairs == [("doanh thu 2016", f"{HEADER}\nnội dung A"), ("doanh thu 2016", "nội dung B")]
assert {c.text for c in out} == {"nội dung A", "nội dung B"}, "chunk.text sau rerank phải là bản GỐC"

ctx.set_enabled(False)
reranker.model = RecordingModel()
reranker.rerank("doanh thu 2016", [make_chunk("a", "nội dung A", HEADER)])
assert reranker.model.pairs == [("doanh thu 2016", "nội dung A")]
print("  ✅ PASS\n")


# ══════════════════════════════════════════════════════════════
# TEST 5: Generator — header CHỈ ở nhãn nguồn, nội dung đoạn giữ nguyên
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 5: Generator._format_context")


class DummyGenerator(BaseGenerator):
    def generate(self, question, chunks, history=None):
        raise NotImplementedError


gen = DummyGenerator()
ctx.set_enabled(True)
formatted = gen._format_context([make_chunk("a", "nội dung A", HEADER)])
assert formatted.startswith(f"--- Tài liệu 1 | Nguồn: f.pdf (Trang 1) | {HEADER} ---\nnội dung A"), formatted
assert formatted.count(HEADER) == 1, "header không được lặp vào nội dung đoạn"

ctx.set_enabled(False)
formatted = gen._format_context([make_chunk("a", "nội dung A", HEADER)])
assert formatted.startswith("--- Tài liệu 1 | Nguồn: f.pdf (Trang 1) ---\nnội dung A"), formatted
assert HEADER not in formatted
print("  ✅ PASS\n")


# ══════════════════════════════════════════════════════════════
# TEST 6: Document model — round-trip + needs_reembed, tài liệu cũ không có trường mới
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 6: Document model")

old = Document.from_dict({"filename": "x.pdf", "original_name": "x.pdf", "file_type": "pdf",
                          "file_size_kb": 1, "uploaded_by": "u1"})
assert old.context_header is None and old.context_header_version == 0 and old.needs_reembed is False

doc = Document("x.pdf", "x.pdf", "pdf", 1, "u1", context_header=HEADER, context_header_source="user",
               context_header_version=2, embedded_header_version=1)
assert doc.needs_reembed is True
again = Document.from_dict(doc.to_dict())
assert (again.context_header, again.context_header_source, again.context_header_version,
        again.embedded_header_version) == (HEADER, "user", 2, 1)
print("  ✅ PASS — tài liệu cũ mặc định không header, không cần embed lại\n")

print("🎉 TẤT CẢ TEST CONTEXT HEADER PASS!")
