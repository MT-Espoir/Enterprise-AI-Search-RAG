"""
Test QueryProcessor (fast path không gọi LLM / SLM path — 2 lời gọi riêng biệt
classify + rewrite, xem query_processor.py / fail-open / unimplemented operations)
và RAGPipeline (bypass_retrieval short-circuit, vector_query truyền đúng) — mock
requests.post + fake retriever/reranker/generator, không cần Ollama/ChromaDB thật.

Chạy: python backend/tests/test_query_processor.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json

import app.core.query.query_processor as qp_module
from app.core.query import QueryProcessor, QueryPlan
from app.core.schemas import RetrievedChunk
from app.core.rag_pipeline import RAGPipeline


class FakeResponse:
    def __init__(self, content: str, status_code: int = 200):
        self._content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return {"message": {"content": self._content}}


def _refuse_llm_call(*a, **kw):
    raise AssertionError("Fast path không được gọi LLM nhưng requests.post đã bị gọi!")


qp = QueryProcessor()

# ══════════════════════════════════════════════════════════════
# TEST 1: Fast path — summary keyword, KHÔNG gọi LLM
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Fast path — summary keyword")

qp_module.requests.post = _refuse_llm_call
plan = qp.process("Hãy tóm tắt tài liệu này giúp tôi")
assert plan.bypass_retrieval is True, f"❌ bypass_retrieval phải True: {plan}"
assert plan.source == "fast_path", f"❌ source phải fast_path: {plan}"
assert plan.query_type == "summary"
print(f"  {plan}")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 2: Fast path — câu rõ ràng, không đại từ/history, KHÔNG gọi LLM
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: Fast path — câu rõ ràng")

plan = qp.process("SensorMQTTListener mặc định subscribe vào MQTT topic nào?")
assert plan.bypass_retrieval is False
assert plan.rewrite is False
assert plan.source == "fast_path"
print(f"  {plan}")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 3: SLM path — câu có đại từ tham chiếu, 2 lời gọi riêng biệt classify + rewrite
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: SLM path — đại từ tham chiếu (2 lời gọi riêng biệt)")


def _dispatch_classify_or_rewrite(*a, **kw):
    """Mock phân biệt call classify vs call rewrite dựa trên system prompt được gửi
    (2 lời gọi khác nhau kể từ Phase 4, xem query_processor.py). So sánh trực tiếp
    với hằng số thật (KHÔNG match chuỗi con cố định) để không vỡ khi nội dung prompt
    đổi — chỉ cần đúng object nào được gửi."""
    messages = kw.get("json", {}).get("messages", [])
    system_content = messages[0]["content"] if messages else ""
    if system_content == qp_module.REWRITE_SYSTEM_INSTRUCTION:
        return FakeResponse(json.dumps({
            "rewrite": True,
            "rewritten_query": "SensorMQTTListener xử lý status_code 2 như thế nào?",
        }))
    assert system_content == qp_module.CLASSIFY_SYSTEM_INSTRUCTION, f"❌ prompt không khớp classify/rewrite nào: {system_content[:80]}"
    return FakeResponse(json.dumps({
        "query_type": "simple",
        "operations": {"expansion": False, "hyde": False, "decomposition": False},
        "retrieval": {"bypass_retrieval": False, "strategy": "hybrid", "top_k": 8},
    }))


qp_module.requests.post = _dispatch_classify_or_rewrite
plan = qp.process("nó xử lý status_code 2 thế nào?", history=[
    {"role": "user", "content": "SensorMQTTListener là gì?"},
])
assert plan.source == "slm"
assert plan.rewrite is True
assert plan.rewritten_query == "SensorMQTTListener xử lý status_code 2 như thế nào?"
assert plan.bypass_retrieval is False
print(f"  {plan}")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 4: SLM fail-open khi request lỗi
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: SLM fail-open")


def _raise(*a, **kw):
    raise ConnectionError("Ollama không chạy")


qp_module.requests.post = _raise
plan = qp.process("nó là cái gì vậy?")  # có "nó" -> cần SLM
assert plan.source == "fallback"
assert plan.bypass_retrieval is False
assert plan.rewrite is False
print(f"  {plan}")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 5: Unimplemented operations — ghi nhận, không crash
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 5: unimplemented_operations")

qp_module.requests.post = lambda *a, **kw: FakeResponse(json.dumps({
    "query_type": "complex",
    "operations": {
        "rewrite": False, "rewritten_query": "",
        "expansion": False, "hyde": False, "decomposition": True,
    },
    "retrieval": {"bypass_retrieval": False, "strategy": "hybrid", "top_k": 8},
}))
plan = qp.process("nó liên quan gì tới cái kia?")
assert plan.unimplemented_operations == ["decomposition"], f"❌ {plan}"
assert plan.bypass_retrieval is False
print(f"  {plan}")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 6: RAGPipeline — bypass_retrieval short-circuit + vector_query truyền đúng
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 6: RAGPipeline integration")


class FakeOps:
    def get_first_chunks_of_doc(self, doc_id, limit=10, acl_where=None):
        return [{"id": "c1", "text": "chunk tóm tắt", "metadata": {"doc_id": doc_id, "filename": "a.pdf"}, "score": 1.0}]


class FakeRetriever:
    def __init__(self):
        self.ops = FakeOps()
        self.last_stats = {"embedding_ms": 1.0, "vector_search_ms": 2.0, "vector_hits": 1}
        self.received_calls = []

    def retrieve(self, question, doc_id=None, history=None, vector_query=None, filters=None, acl_department=None, acl_bypass=False):
        self.received_calls.append({"question": question, "vector_query": vector_query})
        return [RetrievedChunk("c1", "d1", "a.pdf", 1, "nội dung", 0.9, {})]


class FakeReranker:
    def rerank(self, question, chunks):
        return chunks


class FakeGenResult:
    answer = "câu trả lời"
    sources = [{"doc_id": "d1", "filename": "a.pdf", "page": 1}]
    tokens_used = 10
    finish_reason = "stop"


class FakeGenerator:
    def generate(self, question, chunks, history=None):
        return FakeGenResult()


class FakeQueryProcessorBypass:
    def process(self, question, history=None):
        return QueryPlan(query_type="summary", rewrite=False, rewritten_query=None,
                          bypass_retrieval=True, source="fast_path")


retriever = FakeRetriever()
rag = RAGPipeline(retriever=retriever, reranker=FakeReranker(), generator=FakeGenerator(),
                   query_processor=FakeQueryProcessorBypass(), observability_enabled=False)
result = rag.run("tóm tắt tài liệu này", doc_id="doc123")
assert result["answer"] == "câu trả lời"
assert retriever.received_calls == [], f"❌ bypass_retrieval=True không được gọi retriever.retrieve(): {retriever.received_calls}"
print("  bypass_retrieval=True -> KHÔNG gọi retriever.retrieve(): ✅")


class FakeQueryProcessorRewrite:
    def process(self, question, history=None):
        return QueryPlan(query_type="simple", rewrite=True, rewritten_query="REWRITTEN QUERY",
                          bypass_retrieval=False, source="slm")


retriever2 = FakeRetriever()
rag2 = RAGPipeline(retriever=retriever2, reranker=FakeReranker(), generator=FakeGenerator(),
                    query_processor=FakeQueryProcessorRewrite(), observability_enabled=False)
rag2.run("nó là gì?", doc_id=None)
assert retriever2.received_calls[0]["question"] == "nó là gì?"
assert retriever2.received_calls[0]["vector_query"] == "REWRITTEN QUERY", f"❌ {retriever2.received_calls}"
print("  plan.rewrite=True -> vector_query truyền đúng xuống retriever.retrieve(): ✅")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 7: Fast path — câu dài/so sánh KHÔNG đại từ, KHÔNG history vẫn cần SLM
# (Phase 4 mục 2 — trước đây sẽ bị mặc định query_type="simple", không bao giờ
# tới SLM classify, khiến Decomposition không thể kích hoạt cho đúng loại câu này)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 7: Fast path — câu multi-hop dài cần SLM classify")

qp_module.requests.post = lambda *a, **kw: FakeResponse(json.dumps({
    "query_type": "reasoning",
    "operations": {"expansion": False, "hyde": False, "decomposition": False},
    "retrieval": {"bypass_retrieval": False, "strategy": "hybrid", "top_k": 8},
}))
long_question = (
    "Nghị định 145/2020/NĐ-CP hướng dẫn trợ cấp mất việc làm dựa trên căn cứ Điều "
    "nào của Bộ luật Lao động, và điều kiện tối thiểu về thời gian làm việc thường "
    "xuyên để được hưởng là gì?"
)
plan = qp.process(long_question)  # KHÔNG có history, KHÔNG đại từ
assert plan.source == "slm", f"❌ câu dài phải tới SLM classify, không được mặc định fast_path: {plan}"
assert plan.query_type == "reasoning"
print(f"  {plan}")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 8: QueryProcessor.decompose() — tách sub-query hợp lệ + fail-open
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 8: decompose()")

qp_module.requests.post = lambda *a, **kw: FakeResponse(json.dumps({
    "sub_queries": [
        "Nghị định 145/2020/NĐ-CP hướng dẫn trợ cấp mất việc làm dựa trên căn cứ Điều nào của Bộ luật Lao động?",
        "Điều kiện tối thiểu về thời gian làm việc thường xuyên để hưởng trợ cấp mất việc làm là gì?",
    ],
}))
sub_qs = qp.decompose(long_question)
assert len(sub_qs) == 2, f"❌ {sub_qs}"
print(f"  sub_queries hợp lệ: {sub_qs}")

qp_module.requests.post = lambda *a, **kw: FakeResponse(json.dumps({"sub_queries": ["chỉ 1 câu"]}))
sub_qs_fail = qp.decompose(long_question)
assert sub_qs_fail == [], f"❌ decompose phải fail-open (trả []) khi <2 sub_queries: {sub_qs_fail}"
print("  fail-open đúng khi SLM trả <2 sub_queries: ✅")

qp_module.requests.post = _raise
sub_qs_err = qp.decompose(long_question)
assert sub_qs_err == [], f"❌ decompose phải fail-open (trả []) khi lỗi request: {sub_qs_err}"
print("  fail-open đúng khi Ollama lỗi: ✅")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 9: RAGPipeline — Decomposition retrieve riêng từng sub-query rồi merge+dedupe
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 9: RAGPipeline Decomposition (merge nhiều sub-query)")


class FakeQueryProcessorDecompose:
    def process(self, question, history=None):
        return QueryPlan(query_type="reasoning", rewrite=False, rewritten_query=None,
                          bypass_retrieval=False, source="slm")

    def decompose(self, question):
        return ["sub-query 1", "sub-query 2"]


class FakeRetrieverMulti:
    def __init__(self):
        self.ops = FakeOps()
        self.last_stats = {}
        self.received_calls = []

    def retrieve(self, question, doc_id=None, history=None, vector_query=None, filters=None, acl_department=None, acl_bypass=False):
        self.received_calls.append(question)
        if question == "sub-query 1":
            return [RetrievedChunk("c1", "d1", "a.pdf", 1, "nội dung 1", 0.9, {})]
        if question == "sub-query 2":
            # c1 trùng chunk_id với sub-query 1 (điểm thấp hơn -> phải bị loại khi merge)
            # + c2 là chunk mới
            return [
                RetrievedChunk("c1", "d1", "a.pdf", 1, "nội dung 1", 0.5, {}),
                RetrievedChunk("c2", "d1", "b.pdf", 2, "nội dung 2", 0.8, {}),
            ]
        return []


retriever3 = FakeRetrieverMulti()
rag3 = RAGPipeline(retriever=retriever3, reranker=FakeReranker(), generator=FakeGenerator(),
                    query_processor=FakeQueryProcessorDecompose(), observability_enabled=False)
result3 = rag3.run("câu hỏi phức tạp gốc", doc_id=None)
assert retriever3.received_calls == ["sub-query 1", "sub-query 2"], (
    f"❌ phải retrieve riêng từng sub-query: {retriever3.received_calls}"
)
print(f"  retrieve() gọi đúng 1 lần/sub-query: {retriever3.received_calls} ✅")
assert result3["candidates"] == 2, f"❌ phải dedupe c1 (2 sub-query đều trả) còn 2 chunk duy nhất: {result3}"
print("  dedupe theo chunk_id đúng (3 hit thô -> 2 chunk duy nhất): ✅")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 10: Killswitch DECOMPOSITION_ENABLED=False — bỏ qua Decomposition, về flow đơn
# (Phase 4 hậu-kiểm — xem rag_core_quality_roadmap.md mục 6j)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 10: decomposition_enabled=False -> KHÔNG gọi decompose()")


class FakeQueryProcessorDecomposeCounting(FakeQueryProcessorDecompose):
    def __init__(self):
        self.decompose_call_count = 0

    def decompose(self, question):
        self.decompose_call_count += 1
        return ["sub-query 1", "sub-query 2"]


retriever4 = FakeRetrieverMulti()
qp_counting = FakeQueryProcessorDecomposeCounting()
rag4 = RAGPipeline(retriever=retriever4, reranker=FakeReranker(), generator=FakeGenerator(),
                    query_processor=qp_counting, observability_enabled=False,
                    decomposition_enabled=False)
rag4.run("câu hỏi phức tạp gốc", doc_id=None)
assert qp_counting.decompose_call_count == 0, (
    f"❌ decomposition_enabled=False phải bỏ qua decompose() hoàn toàn: gọi {qp_counting.decompose_call_count} lần"
)
assert retriever4.received_calls == ["câu hỏi phức tạp gốc"], (
    f"❌ phải retrieve 1 lần bằng câu hỏi gốc (flow đơn), không phải sub-query: {retriever4.received_calls}"
)
print("  decompose() KHÔNG được gọi khi tắt killswitch: ✅")
print(f"  retrieve() dùng lại flow đơn (câu hỏi gốc): {retriever4.received_calls} ✅")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 11: Tương tác Rewrite + Decomposition cùng kích hoạt trong 1 câu hỏi
# (câu vừa có đại từ/tham chiếu tới history VỪA multi-hop/so sánh — trước đây
# CHƯA có test nào phủ tình huống này, xem rag_core_quality_roadmap.md mục 6j)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 11: Rewrite + Decomposition cùng kích hoạt — decompose() phải nhận câu ĐÃ rewrite")


class FakeQueryProcessorRewriteAndDecompose:
    def __init__(self):
        self.decompose_received_question = None

    def process(self, question, history=None):
        return QueryPlan(
            query_type="comparison", rewrite=True,
            rewritten_query="So sánh trợ cấp thôi việc với tiền lương làm thêm giờ (câu ĐÃ rewrite, không còn đại từ)",
            bypass_retrieval=False, source="slm",
        )

    def decompose(self, question):
        self.decompose_received_question = question
        return ["sub-query A (từ câu đã rewrite)", "sub-query B (từ câu đã rewrite)"]


retriever5 = FakeRetrieverMulti()
qp_both = FakeQueryProcessorRewriteAndDecompose()
rag5 = RAGPipeline(retriever=retriever5, reranker=FakeReranker(), generator=FakeGenerator(),
                    query_processor=qp_both, observability_enabled=False)
rag5.run("So sánh khoản đó với tiền lương làm thêm giờ?", doc_id=None,
         history=[{"role": "user", "content": "Nghị định 145 nói gì về trợ cấp thôi việc?"}])
assert qp_both.decompose_received_question == (
    "So sánh trợ cấp thôi việc với tiền lương làm thêm giờ (câu ĐÃ rewrite, không còn đại từ)"
), f"❌ decompose() phải nhận effective_question (đã rewrite), không phải câu gốc còn đại từ: {qp_both.decompose_received_question!r}"
print(f"  decompose() nhận đúng câu ĐÃ rewrite (không phải câu gốc còn đại từ): ✅")
print("  ✅ PASS\n")

print("=" * 60)
print("🎉 TẤT CẢ TESTS PASS!")
