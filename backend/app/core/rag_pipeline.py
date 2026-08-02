import logging
import time
import uuid

from .retriever import Retriever, RetrievedChunk
from .reranker import Reranker
from .generator import Generator, GenerationResult
from .query_processor import QueryProcessor
from .observability.request_tracer import log_rag_request
from .guardrails import sanitize_chunks, check_output
from ..vectorstore.operations import build_acl_where

logger = logging.getLogger(__name__)

_MULTI_RETRIEVE_CAP = 15

def _empty_guardrail() -> dict:
    """Guardrail summary mặc định cho các nhánh early-return (NO_DOC/NO_DOCS/
    EMPTY_DOC) — không có chunk/answer nào để sanitize/check, nhưng key
    "guardrail" vẫn phải luôn có mặt để chat.py gọi .get("guardrail") không
    bao giờ trả None bất ngờ."""
    return {"sanitized_chunks": 0, "citation_warning": False, "unverified_citations": []}


class RAGPipeline:
    """
    Điều phối toàn bộ luồng hỏi đáp:
    - QueryProcessor lập QueryPlan (bypass_retrieval/rewrite/query_type/...) thay
      cho IntentRouter cũ (xem query_processor.py — hợp nhất IntentRouter +
      QueryAnalyzer + QueryRewriter).
    - plan.bypass_retrieval=True (vd tóm tắt): bỏ qua Retrieval + Reranking,
      lấy thẳng 10 chunk đầu của document.
    - plan.query_type in ("complex", "comparison", "reasoning"): thử Decomposition
      (QueryProcessor.decompose()) — tách thành sub-query, retrieve riêng từng
      sub-query rồi merge trước khi Rerank. Fail-open về Retrieve đơn giản nếu
      decompose không tách được (xem rag_core_quality_roadmap.md mục 6i).
    - Ngược lại: Retrieve (top 10) → Rerank (top 3) → Generate.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        query_processor: QueryProcessor = None,
        observability_enabled: bool = True,
        decomposition_enabled: bool = True,
        retrieval_guardrails_enabled: bool = True,
        output_guardrails_enabled: bool = True,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.query_processor = query_processor or QueryProcessor()
        self.observability_enabled = observability_enabled
        self.decomposition_enabled = decomposition_enabled
        # Killswitch RIÊNG khỏi query_processor.slm_enabled — xem config.py
        # DECOMPOSITION_ENABLED (rag_core_quality_roadmap.md mục 6j).
        self.retrieval_guardrails_enabled = retrieval_guardrails_enabled
        self.output_guardrails_enabled = output_guardrails_enabled
        # Tier 1 Security Guardrails — killswitch RIÊNG cho từng layer (xem
        # config.py RETRIEVAL_GUARDRAILS_ENABLED/OUTPUT_GUARDRAILS_ENABLED),
        # fail-open tuyệt đối bên trong sanitize_chunks()/check_output().

    def run(self, question: str, doc_id: str = None, history: list[dict] = None, request_id: str = None, filters: dict = None,
            acl_department: str = None, acl_bypass: bool = False) -> dict:
        """
        Chạy pipeline RAG đầy đủ.

        filters: điều kiện metadata BỔ SUNG ngoài doc_id (vd {"document_status":
        "hieu_luc", "document_type": "luat"}) — Phase 4 mục 4, xem
        rag_core_quality_roadmap.md mục 6l. Áp dụng cho nhánh QA (Retrieve/
        Decomposition) — KHÔNG áp dụng cho nhánh tóm tắt (bypass_retrieval), vì
        đó luôn thao tác trên 1 doc_id cụ thể user đã chọn sẵn.

        acl_department/acl_bypass: Document-level ACL, server-derived từ JWT
        trong chat.py — KHÁC với `filters` (client-controlled), áp dụng cho
        CẢ 2 nhánh (bypass VÀ QA) vì cả 2 đều có thể lộ nội dung tài liệu bị
        hạn chế nếu bỏ sót.
        """
        request_id = request_id or str(uuid.uuid4())
        t_start = time.perf_counter()

        logger.info(f"RAG query: '{question[:80]}' | doc_filter={doc_id} | history={len(history or [])} msgs")

        # ── Bước 0: Query Processing (fast path hoặc SLM) ─────
        t0 = time.perf_counter()
        plan = self.query_processor.process(question, history=history)
        preprocessing_ms = round((time.perf_counter() - t0) * 1000, 2)

        candidates: list[RetrievedChunk] = []
        top_chunks: list[RetrievedChunk] = []
        retrieval_ms = None
        rerank_ms = None
        retrieval_stats: dict = {}

        if plan.bypass_retrieval:
            logger.info("Luồng TÓM TẮT (bypass_retrieval): Bỏ qua Vector Search & Reranking.")
            if not doc_id:
                self._trace(request_id, question, plan, preprocessing_ms, None, None, None, {}, 0)
                return {
                    "answer":      "Vui lòng chỉ định rõ một tài liệu cụ thể để tôi có thể tóm tắt nhé.",
                    "sources":     [],
                    "tokens_used": 0,
                    "candidates":  0,
                    "finish_reason": "NO_DOC",
                    "guardrail":   _empty_guardrail(),
                }

            # Lấy 10 chunk đầu tiên của doc_id từ VectorDB (bỏ qua semantic search)
            # ACL: nếu phòng ban không khớp, get_first_chunks_of_doc trả rỗng ->
            # rơi vào nhánh "if not top_chunks" bên dưới, response giống hệt tài
            # liệu trống/không tồn tại (không lộ thông tin tài liệu bị chặn).
            acl_where = None if acl_bypass else build_acl_where(acl_department)
            raw_chunks = self.retriever.ops.get_first_chunks_of_doc(doc_id, limit=10, acl_where=acl_where)
            for item in raw_chunks:
                meta = item.get("metadata", {})
                candidates.append(RetrievedChunk(
                    chunk_id = item["id"],
                    doc_id   = meta.get("doc_id", ""),
                    filename = meta.get("filename", ""),
                    page     = meta.get("page_num"),
                    text     = item["text"],
                    score    = item["score"],
                    metadata = meta,
                ))

            top_chunks = candidates  # Không chạy Reranking cho luồng bypass

        else:
            # ── Luồng QA: Retrieve + Rerank bình thường (hoặc Decomposition) ────
            effective_question = plan.rewritten_query if plan.rewrite else question
            sub_queries: list[str] = []
            if self.decomposition_enabled and plan.query_type in ("complex", "comparison", "reasoning"):
                sub_queries = self.query_processor.decompose(effective_question)

            t0 = time.perf_counter()
            if len(sub_queries) >= 2:
                logger.info(
                    f"Luồng DECOMPOSITION ({plan.query_type}): tách '{question[:60]}' "
                    f"thành {len(sub_queries)} sub-query: {sub_queries}"
                )
                candidates = self._retrieve_multi(sub_queries, doc_id=doc_id, filters=filters, acl_department=acl_department, acl_bypass=acl_bypass)
            else:
                logger.info("Luồng HỎI ĐÁP (QA): Chạy Retrieval -> Rerank.")
                vector_query = plan.rewritten_query if plan.rewrite else None
                candidates = self.retriever.retrieve(question, doc_id=doc_id, history=history, vector_query=vector_query, filters=filters, acl_department=acl_department, acl_bypass=acl_bypass)
            retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)
            retrieval_stats = getattr(self.retriever, "last_stats", {}) or {}

            if not candidates:
                logger.warning("Không tìm thấy chunks nào liên quan.")
                self._trace(request_id, question, plan, preprocessing_ms, retrieval_ms, None, None, retrieval_stats, 0)
                return {
                    "answer":      "Tôi không tìm thấy tài liệu nào liên quan đến câu hỏi của bạn.",
                    "sources":     [],
                    "tokens_used": 0,
                    "candidates":  0,
                    "finish_reason": "NO_DOCS",
                    "guardrail":   _empty_guardrail(),
                }

            t0 = time.perf_counter()
            top_chunks = self.reranker.rerank(question, candidates)
            rerank_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(f"Reranked: {len(candidates)} → {len(top_chunks)} chunks")

        if not top_chunks:
            self._trace(request_id, question, plan, preprocessing_ms, retrieval_ms, rerank_ms, None, retrieval_stats, len(candidates))
            return {
                "answer":      "Tài liệu này hiện đang trống hoặc chưa được xử lý văn bản.",
                "sources":     [],
                "tokens_used": 0,
                "candidates":  0,
                "finish_reason": "EMPTY_DOC",
                "guardrail":   _empty_guardrail(),
            }

        # Retrieval Guardrail (Tier 1, Layer 4) 
        # Điểm neo DUY NHẤT bao phủ CẢ nhánh bypass (tóm tắt) lẫn nhánh QA/
        # Decomposition — cả 2 đều hội tụ vào top_chunks tới đây. Fail-open
        # tuyệt đối bên trong sanitize_chunks(), nhưng vẫn bọc try/except ở
        # đây để 1 lỗi ngoài dự kiến không làm rớt cả request.
        sanitize_report = None
        if self.retrieval_guardrails_enabled:
            try:
                top_chunks, sanitize_report = sanitize_chunks(top_chunks)
            except Exception as exc:
                logger.warning(f"Retrieval guardrail lỗi ({exc}) — bỏ qua, dùng chunks gốc.")

        #Bước cuối: Generate 
        t0 = time.perf_counter()
        result: GenerationResult = self.generator.generate(question, top_chunks, history=history)
        generation_ms = round((time.perf_counter() - t0) * 1000, 2)

        #Output Guardrail (Tier 1, Layer 7) 
        # Chỉ cảnh báo, KHÔNG chặn (quyết định đã chốt) — xem guardrails/output_guard.py.
        output_check = None
        if self.output_guardrails_enabled:
            try:
                output_check = check_output(result.answer, top_chunks)
            except Exception as exc:
                logger.warning(f"Output guardrail lỗi ({exc}) — bỏ qua, không gắn cảnh báo.")

        guardrail_summary = {
            "sanitized_chunks": sanitize_report.chunks_sanitized if sanitize_report else 0,
            "citation_warning": output_check.warning if output_check else False,
            "unverified_citations": output_check.unverified_citations if output_check else [],
        }

        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        self._trace(
            request_id, question, plan, preprocessing_ms, retrieval_ms, rerank_ms, generation_ms,
            retrieval_stats, len(candidates), reranked_hits=len(top_chunks), total_ms=total_ms,
            guardrail_summary=guardrail_summary,
        )

        return {
            "answer":        result.answer,
            "sources":       result.sources,
            "tokens_used":   result.tokens_used,
            "candidates":    len(candidates),
            "finish_reason": result.finish_reason,
            "guardrail":     guardrail_summary,
        }

    def _retrieve_multi(self, sub_queries: list[str], doc_id: str = None, filters: dict = None,
                         acl_department: str = None, acl_bypass: bool = False) -> list[RetrievedChunk]:
        """Retrieve riêng từng sub-query (Decomposition/Multi-Query, xem docstring
        class) rồi merge + dedupe theo chunk_id — giữ điểm cao nhất nếu 1 chunk
        được nhiều sub-query cùng trả về. Cap kết quả ở _MULTI_RETRIEVE_CAP trước
        khi đưa vào Reranker. `filters`/ACL áp dụng như nhau cho MỌI sub-query."""
        merged: dict[str, RetrievedChunk] = {}
        for sub_q in sub_queries:
            hits = self.retriever.retrieve(sub_q, doc_id=doc_id, filters=filters, acl_department=acl_department, acl_bypass=acl_bypass)
            for chunk in hits:
                existing = merged.get(chunk.chunk_id)
                if existing is None or chunk.score > existing.score:
                    merged[chunk.chunk_id] = chunk
        ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        return ranked[:_MULTI_RETRIEVE_CAP]

    def _trace(
        self, request_id, question, plan, preprocessing_ms, retrieval_ms, rerank_ms, generation_ms,
        retrieval_stats, candidates_count, reranked_hits=None, total_ms=None, guardrail_summary=None,
    ) -> None:
        log_rag_request(
            request_id=request_id,
            question=question,
            guardrail=guardrail_summary,
            query_plan={
                "query_type": plan.query_type,
                "strategy_hint": plan.strategy_hint,
                "rewrite": plan.rewrite,
                "bypass_retrieval": plan.bypass_retrieval,
                "unimplemented_operations": plan.unimplemented_operations,
                "source": plan.source,
            },
            latency={
                "preprocessing_ms": preprocessing_ms,
                "retrieval_ms": retrieval_ms,
                "embedding_ms": retrieval_stats.get("embedding_ms"),
                "vector_search_ms": retrieval_stats.get("vector_search_ms"),
                "bm25_search_ms": retrieval_stats.get("bm25_search_ms"),
                "rerank_ms": rerank_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
            retrieval={
                "vector_hits": retrieval_stats.get("vector_hits"),
                "bm25_hits": retrieval_stats.get("bm25_hits"),
                "fusion_hits": retrieval_stats.get("fusion_hits"),
                "candidates": candidates_count,
                "reranked_hits": reranked_hits,
            },
            enabled=self.observability_enabled,
        )
