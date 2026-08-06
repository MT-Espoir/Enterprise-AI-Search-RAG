"""Nhóm TRUY HỒI (retrieval): tìm chunk liên quan từ kho dữ liệu.

Gồm 2 nhánh tìm kiếm (vector qua ChromaDB, keyword qua BM25) + hợp nhất RRF,
factory chọn chiến lược theo config, và cơ chế đồng bộ BM25 giữa các worker.

Bước XẾP HẠNG LẠI (rerank) KHÔNG thuộc nhóm này — xem core/ranking/.
"""
from .base_retriever import BaseRetriever
from .retriever import Retriever
from .hybrid_retriever import HybridRetriever
from .retriever_factory import build_retriever
from .bm25_index import BM25Index, get_bm25_index, simple_tokenize
from .bm25_sync import mark_bm25_dirty, start_bm25_sync_worker

__all__ = [
    "BaseRetriever", "Retriever", "HybridRetriever", "build_retriever",
    "BM25Index", "get_bm25_index", "simple_tokenize",
    "mark_bm25_dirty", "start_bm25_sync_worker",
]
