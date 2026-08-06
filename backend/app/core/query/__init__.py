"""Nhóm XỬ LÝ TRUY VẤN (query processing): phân tích câu hỏi TRƯỚC khi truy hồi.

Hợp nhất phân loại ý định, viết lại câu hỏi tỉnh lược, và tách câu hỏi phức tạp
thành các câu con — đầu ra là một QueryPlan cho RAGPipeline thực thi.
"""
from .query_processor import QueryProcessor, QueryPlan

__all__ = ["QueryProcessor", "QueryPlan"]
