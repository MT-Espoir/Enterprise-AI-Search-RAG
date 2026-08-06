import time

from .base_retriever import BaseRetriever
from .bm25_index import BM25Index
from ..schemas import RetrievedChunk
from .retriever import Retriever


class HybridRetriever(BaseRetriever):
    """
    Hợp nhất Vector search (ChromaDB cosine) + BM25 keyword search bằng
    Reciprocal Rank Fusion (RRF). Vector search bù ngữ nghĩa, BM25 bù
    exact-match từ khóa/mã số mà bi-encoder dễ bỏ sót ở corpus lớn.

    bm25_index là constructor-injected — class này không tự biết/không quan tâm
    instance đến từ đâu (global getter hay tạo riêng cho test), giữ interface
    thuần dependency-injection để dễ mock/test độc lập.
    """

    def __init__(
        self,
        ops,
        embedder,
        bm25_index: BM25Index,
        vector_pool_size: int = 20,
        bm25_pool_size: int = 20,
        rrf_k: int = 60,
        top_k: int = 10,
    ):
        self.ops = ops  # bắt buộc: RAGPipeline SUMMARY flow đọc self.retriever.ops
        self._vector_retriever = Retriever(ops=ops, embedder=embedder, top_k=vector_pool_size)
        self._bm25_index = bm25_index
        self.bm25_pool_size = bm25_pool_size
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.last_stats: dict = {}

    def retrieve(
        self,
        question: str,
        doc_id: str = None,
        history: list[dict] = None,
        vector_query: str = None,
        filters: dict = None,
        acl_department: str = None,
        acl_bypass: bool = False,
    ) -> list[RetrievedChunk]:
        # BM25 luôn dùng `question` GỐC (exact-match từ khóa, nhạy với đổi từ ngữ —
        # xem roadmap mục 4c). `vector_query` (nếu có, từ QueryPlan.rewritten_query) chỉ
        # phục vụ nhánh vector, không bao giờ thay thế question cho BM25. `filters`
        # (document_status/document_type — Phase 4 mục 4) áp dụng cho CẢ 2 nhánh như nhau,
        # khác với vector_query/doc_id — đây là điều kiện thu hẹp phạm vi tài liệu, không
        # phải cách diễn đạt câu hỏi. `acl_department`/`acl_bypass` (Document-level ACL)
        # cũng áp dụng CẢ 2 nhánh như nhau — tránh lộ chunk qua BM25 dù vector đã chặn đúng.
        vector_hits = self._vector_retriever.retrieve(question, doc_id=doc_id, vector_query=vector_query, filters=filters, acl_department=acl_department, acl_bypass=acl_bypass)
        t0 = time.perf_counter()
        bm25_hits = self._bm25_index.search(question, top_k=self.bm25_pool_size, doc_id=doc_id, filters=filters, acl_department=acl_department, acl_bypass=acl_bypass)
        bm25_search_ms = round((time.perf_counter() - t0) * 1000, 2)

        merged: dict[str, RetrievedChunk] = {}
        rrf_accum: dict[str, float] = {}

        for rank, chunk in enumerate(vector_hits, start=1):
            chunk_id = chunk.chunk_id
            merged[chunk_id] = chunk
            chunk.metadata["vector_rank"] = rank
            chunk.metadata["vector_score"] = chunk.score
            rrf_accum[chunk_id] = rrf_accum.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, hit in enumerate(bm25_hits, start=1):
            chunk_id = hit["id"]
            if chunk_id not in merged:
                meta = hit["metadata"]
                merged[chunk_id] = RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    filename=meta.get("filename", ""),
                    page=meta.get("page_num"),
                    text=hit["text"],
                    score=0.0,
                    metadata=meta,
                )
            merged[chunk_id].metadata["bm25_rank"] = rank
            merged[chunk_id].metadata["bm25_score"] = hit["score"]
            rrf_accum[chunk_id] = rrf_accum.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        for chunk_id, chunk in merged.items():
            chunk.score = rrf_accum[chunk_id]
            chunk.metadata["rrf_score"] = rrf_accum[chunk_id]

        ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        result = ranked[: self.top_k]

        self.last_stats = {
            **self._vector_retriever.last_stats,  # embedding_ms, vector_search_ms, vector_hits
            "bm25_search_ms": bm25_search_ms,
            "bm25_hits": len(bm25_hits),
            "fusion_hits": len(result),
        }

        return result
