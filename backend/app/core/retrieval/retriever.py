import time
from ...vectorstore.operations import VectorStoreOps, build_chroma_where, build_acl_where, combine_where
from ...ingestion.embedder.base_embedder import BaseEmbedder
from .base_retriever import BaseRetriever
# RetrievedChunk nay định nghĩa ở core/schemas.py (module lá dùng chung) — re-export
# ở đây để mọi import cũ `from .retriever import RetrievedChunk` vẫn hoạt động.
from ..schemas import RetrievedChunk


class Retriever(BaseRetriever):
    """
    Bước 1 trong pipeline tìm kiếm: Bi-Encoder retrieval.
    Nhận câu hỏi → embed → query ChromaDB → trả về top_k candidates.
    """

    def __init__(self, ops: VectorStoreOps, embedder: BaseEmbedder, top_k: int = 20):
        self.ops      = ops
        self.embedder = embedder
        self.top_k    = top_k
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
        """
        Tìm kiếm các chunks liên quan đến câu hỏi.

        Args:
            question:     Câu hỏi gốc từ người dùng.
            doc_id:       (Tùy chọn) Giới hạn tìm kiếm trong 1 document cụ thể.
            history:      (Tùy chọn) Không dùng ở đây — chỉ có ý nghĩa với QueryProcessor/RAGPipeline.
                          Giữ trong signature để mọi BaseRetriever có thể hoán đổi cho nhau.
            vector_query: (Tùy chọn) Câu truy vấn đã viết lại/mở rộng để embed thay cho `question`
                          (vd từ QueryPlan.rewritten_query). Mặc định dùng `question` nếu không truyền.
            filters:      (Tùy chọn) Điều kiện metadata BỔ SUNG ngoài doc_id (vd
                          {"document_status": "hieu_luc"}) — Phase 4 mục 4, xem
                          rag_core_quality_roadmap.md mục 6l. Merge với doc_id thành 1 where-clause.
            acl_department/acl_bypass: Document-level ACL, server-derived từ JWT
                          (xem chat.py) — TÁCH BIỆT hoàn toàn khỏi `filters` (client-controlled)
                          để tránh spoof qua request body. acl_bypass=True (admin) bỏ qua
                          hoàn toàn mệnh đề ACL.

        Returns:
            list[RetrievedChunk] sắp xếp theo score giảm dần.

        Side effect: cập nhật self.last_stats (embedding_ms/vector_search_ms/vector_hits) —
        side-channel cho observability, không đổi return type/signature của bất kỳ ai gọi.
        """
        # Bước 1: Chuyển câu hỏi thành vector — ưu tiên vector_query nếu có (query expansion)
        t0 = time.perf_counter()
        query_vector = self.embedder.embed_query(vector_query or question)
        t1 = time.perf_counter()

        # Bước 2: Tìm kiếm trong ChromaDB
        combined_filters = dict(filters or {})
        if doc_id:
            combined_filters["doc_id"] = doc_id
        base_where = build_chroma_where(combined_filters)
        acl_where = None if acl_bypass else build_acl_where(acl_department)
        filter_meta = combine_where(base_where, acl_where)
        raw_results = self.ops.query(
            query_embedding=query_vector,
            n_results=self.top_k,
            filter_metadata=filter_meta,
        )
        t2 = time.perf_counter()

        # Bước 3: Map sang RetrievedChunk dataclass
        chunks = []
        for item in raw_results:
            meta = item.get("metadata", {})
            chunks.append(RetrievedChunk(
                chunk_id = item["id"],
                doc_id   = meta.get("doc_id", ""),
                filename = meta.get("filename", ""),
                page     = meta.get("page_num"),
                text     = item["text"],
                score    = item["score"],
                metadata = meta,
            ))

        self.last_stats = {
            "embedding_ms": round((t1 - t0) * 1000, 2),
            "vector_search_ms": round((t2 - t1) * 1000, 2),
            "vector_hits": len(chunks),
        }

        return chunks
