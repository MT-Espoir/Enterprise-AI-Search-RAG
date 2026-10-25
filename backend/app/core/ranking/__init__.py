"""Nhóm XẾP HẠNG LẠI (ranking): sắp xếp lại các chunk ứng viên do retrieval trả về.

Tách riêng khỏi retrieval vì đây là tầng khác biệt về bản chất: retrieval dùng
bi-encoder/BM25 để LỌC NHANH trên toàn corpus, còn ranking dùng cross-encoder
(đọc đồng thời câu hỏi + chunk) để xếp hạng CHÍNH XÁC trên một tập ứng viên nhỏ.
Chi phí và mô hình khác hẳn nhau, có thể thay thế/tắt độc lập.
"""
from .reranker import Reranker
from .selection import dedupe_identical, select_diverse

__all__ = ["Reranker", "dedupe_identical", "select_diverse"]
