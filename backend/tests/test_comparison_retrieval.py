"""
Test nhánh câu SO SÁNH / tổng hợp nhiều tài liệu (multi_doc_selection_plan.md, M2):
  - select_diverse(): mỗi tài liệu 1 chunk điểm cao nhất, gộp chunk trùng nguyên văn, lấp khi ít tài liệu
  - is_comparison_query(): chỉ `comparison` của SLM hoặc regex so sánh — KHÔNG reasoning/complex
  - Reranker.rerank(top_k=...) ghi đè số chunk trả về
  - RAGPipeline: killswitch tắt → top-k theo điểm như cũ; bật + câu so sánh → đa dạng tài liệu; bật + câu thường →
    y nguyên bản tắt; plan.diverse_selection phản ánh đúng
  - QueryProcessor.decompose(): prompt tách THEO đối tượng chỉ khi decompose_by_entity bật
  - config: 2 killswitch mặc định TẮT
Không cần Ollama / ChromaDB / model thật (fake + mock requests.post).

Chạy: python backend/tests/test_comparison_retrieval.py
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import app.core.query.query_processor as qp_module
from app.config import BaseConfig
from app.core.query import QueryPlan, QueryProcessor, is_comparison_query
from app.core.query.prompts import QUERY_DECOMPOSE_BY_ENTITY_SYSTEM_INSTRUCTION, QUERY_DECOMPOSE_SYSTEM_INSTRUCTION
from app.core.rag_pipeline import RAGPipeline
from app.core.ranking import Reranker, dedupe_identical, select_diverse
from app.core.schemas import RetrievedChunk


def chunk(cid, doc, text=None):
    return RetrievedChunk(cid, doc, f"{doc}.pdf", 1, text or f"nội dung {cid}", 0.5, {})


def ids(chunks):
    return [c.chunk_id for c in chunks]


# Thứ tự = điểm giảm dần: 3 chunk đầu cùng tài liệu A (đúng kiểu dồn vào 1 tài liệu), x1 trùng nguyên văn a1
A1, X1, A2, A3, B1, C1 = chunk("a1", "A"), chunk("x1", "X", "nội dung a1"), chunk("a2", "A"), chunk("a3", "A"), chunk("b1", "B"), chunk("c1", "C")
RANKED = [A1, X1, A2, A3, B1, C1]

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: select_diverse / dedupe_identical")
assert ids(dedupe_identical(RANKED)) == ["a1", "a2", "a3", "b1", "c1"], "x1 trùng nguyên văn a1 phải bị gộp"
assert ids(select_diverse(RANKED, 3)) == ["a1", "b1", "c1"], ids(select_diverse(RANKED, 3))
assert ids(select_diverse([A1, A2, B1], 3)) == ["a1", "b1", "a2"], "ít tài liệu hơn k → lấp theo điểm"
assert ids(select_diverse([A1, A2], 3)) == ["a1", "a2"], "ít chunk hơn k → trả hết"
assert ids(select_diverse(RANKED, 1)) == ["a1"]
assert select_diverse([], 3) == []
assert ids(RANKED) == ["a1", "x1", "a2", "a3", "b1", "c1"], "không được sửa danh sách đầu vào"
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: is_comparison_query — chỉ comparison hoặc regex so sánh")
plan = lambda t: QueryPlan(query_type=t, rewrite=False, rewritten_query=None, bypass_retrieval=False)
assert is_comparison_query("So sánh lợi nhuận Vinaconex năm 2019 và 2020")
assert is_comparison_query("Hai mẫu điều lệ khác nhau thế nào?")
assert is_comparison_query("Doanh thu năm nay so với năm trước?")
assert is_comparison_query("Thẩm quyền ký đề nghị thay đổi người đại diện", plan("comparison")), "SLM comparison phải kích hoạt"
for t in ("reasoning", "complex", "simple", "summary"):
    assert not is_comparison_query("Ai có quyền bổ nhiệm Chủ tịch Hội đồng thành viên?", plan(t)), t
assert not is_comparison_query("Ai có quyền bổ nhiệm Chủ tịch Hội đồng thành viên?")
assert not is_comparison_query("")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: Reranker.rerank(top_k=...) ghi đè số chunk trả về")


class FakeCrossEncoder:
    def predict(self, pairs):
        return [1.0 - i * 0.1 for i in range(len(pairs))]


rr = object.__new__(Reranker)  # không nạp model thật
rr.model, rr.top_k = FakeCrossEncoder(), 3
pool = [chunk(f"p{i}", f"D{i}") for i in range(6)]
assert len(rr.rerank("q", pool)) == 3
assert len(rr.rerank("q", pool, top_k=6)) == 6
assert len(rr.rerank("q", pool, top_k=None)) == 3
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: RAGPipeline — killswitch, câu so sánh / câu thường")


class FakeRetriever:
    last_stats = {}

    def retrieve(self, question, **kwargs):
        return [chunk(c.chunk_id, c.doc_id, c.text) for c in RANKED]


class FakeReranker:
    """Giữ thứ tự RANKED làm thứ tự điểm; tôn trọng top_k như Reranker thật."""
    top_k = 3

    def __init__(self):
        self.calls = []

    def rerank(self, question, chunks, top_k=None):
        self.calls.append(top_k)
        return list(chunks)[:self.top_k if top_k is None else top_k]


class FakeGenResult:
    answer = "câu trả lời"
    sources = []
    tokens_used = 1
    finish_reason = "stop"


class FakeGenerator:
    def __init__(self):
        self.received = None

    def generate(self, question, chunks, history=None):
        self.received = ids(chunks)
        return FakeGenResult()


class FakeQP:
    def __init__(self, query_type="simple"):
        self.query_type = query_type
        self.last_plan = None

    def process(self, question, history=None):
        self.last_plan = QueryPlan(query_type=self.query_type, rewrite=False, rewritten_query=None, bypass_retrieval=False)
        return self.last_plan

    def decompose(self, question):
        return []  # fail-open → retrieve đơn giản


def run(question, diversity, query_type="simple"):
    gen, reranker, qp = FakeGenerator(), FakeReranker(), FakeQP(query_type)
    RAGPipeline(retriever=FakeRetriever(), reranker=reranker, generator=gen, query_processor=qp,
                observability_enabled=False, diversity_enabled=diversity).run(question)
    return gen.received, qp.last_plan.diverse_selection, reranker.calls


COMPARE = "So sánh chi phí nhân công của hai năm"
PLAIN = "Chi phí nhân công là bao nhiêu?"
off = run(COMPARE, diversity=False)
assert off == (["a1", "x1", "a2"], False, [None]), off
on = run(COMPARE, diversity=True)
assert on[0] == ["a1", "b1", "c1"] and on[1] is True and on[2] == [len(RANKED)], on
plain_off, plain_on = run(PLAIN, diversity=False), run(PLAIN, diversity=True)
assert plain_on == plain_off == (["a1", "x1", "a2"], False, [None]), "câu thường: bật hay tắt phải y nguyên"
assert run(PLAIN, diversity=True, query_type="comparison")[1] is True, "SLM comparison (không có từ so sánh) phải kích hoạt"
assert run(PLAIN, diversity=True, query_type="reasoning")[1] is False, "reasoning KHÔNG kích hoạt"
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 5: QueryProcessor.decompose — prompt tách theo đối tượng chỉ khi bật")
sent = []


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": json.dumps({"sub_queries": ["câu con 1", "câu con 2"]})}}


def fake_post(url, json=None, timeout=None):
    sent.append(json["messages"][0]["content"])
    return FakeResponse()


original_post = qp_module.requests.post
qp_module.requests.post = fake_post
try:
    assert QueryProcessor().decompose("So sánh A và B") == ["câu con 1", "câu con 2"]
    assert QueryProcessor(decompose_by_entity=True).decompose("So sánh A và B") == ["câu con 1", "câu con 2"]
finally:
    qp_module.requests.post = original_post
assert sent == [QUERY_DECOMPOSE_SYSTEM_INSTRUCTION, QUERY_DECOMPOSE_BY_ENTITY_SYSTEM_INSTRUCTION], sent
assert "ĐÚNG MỘT đối tượng" in QUERY_DECOMPOSE_BY_ENTITY_SYSTEM_INSTRUCTION
assert QueryProcessor().decompose_by_entity is False
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 6: killswitch mặc định TẮT")
if not os.getenv("RERANK_DIVERSITY_ENABLED"):
    assert BaseConfig.RERANK_DIVERSITY_ENABLED is False
if not os.getenv("DECOMPOSE_BY_ENTITY_ENABLED"):
    assert BaseConfig.DECOMPOSE_BY_ENTITY_ENABLED is False
assert RAGPipeline(retriever=FakeRetriever(), reranker=FakeReranker(), generator=FakeGenerator(),
                   query_processor=FakeQP()).diversity_enabled is False
print("  ✅ PASS\n")

print("🎉 TẤT CẢ TEST CÂU SO SÁNH PASS!")
