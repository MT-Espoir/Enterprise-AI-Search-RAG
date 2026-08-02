from .base_retriever import BaseRetriever
from .bm25_index import BM25Index
from .hybrid_retriever import HybridRetriever
from .retriever import Retriever


def build_retriever(strategy: str, ops, embedder, bm25_index: BM25Index = None, **kwargs) -> BaseRetriever:
    """
    Factory tập trung việc chọn retrieval strategy — tránh lặp if/else ở
    từng call site (chat.py, run_full_benchmark.py).

    strategy:
      - "hybrid": Vector + BM25 (RRF fusion). Bắt buộc truyền bm25_index.
      - "vector": Vector-only (hành vi cũ, không đổi).
    """
    if strategy == "hybrid":
        if bm25_index is None:
            raise ValueError("strategy='hybrid' cần truyền bm25_index")
        return HybridRetriever(
            ops=ops,
            embedder=embedder,
            bm25_index=bm25_index,
            vector_pool_size=kwargs.get("vector_pool_size", 20),
            bm25_pool_size=kwargs.get("bm25_pool_size", 20),
            rrf_k=kwargs.get("rrf_k", 60),
            top_k=kwargs.get("top_k", 10),
        )

    if strategy == "vector":
        return Retriever(ops=ops, embedder=embedder, top_k=kwargs.get("top_k", 10))

    raise ValueError(f"Unknown retrieval strategy: {strategy}")
