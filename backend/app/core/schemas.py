"""Kiểu dữ liệu dùng CHUNG giữa các nhóm trong core (retrieval / ranking /
generation / guardrails / pipeline).

Vì sao tách riêng module này: `RetrievedChunk` trước đây định nghĩa trong
`retriever.py` nhưng được 10 module import — kể cả generator và guardrails.
Khi core được tách theo trách nhiệm (retrieval/, ranking/, generation/...),
việc generation phải import ngược từ retrieval chỉ để lấy 1 dataclass sẽ tạo
coupling chéo và rủi ro vòng lặp import.

Module này là LÁ: KHÔNG import bất kỳ module nào khác trong core — nhờ đó mọi
nhóm đều có thể phụ thuộc vào nó mà không tạo chu trình.
"""
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """Đại diện cho một chunk đã được tìm kiếm và có điểm similarity."""
    chunk_id : str
    doc_id   : str
    filename : str
    page     : int | None
    text     : str
    score    : float          # Cosine similarity (0~1) hoặc RRF score tùy strategy, càng cao càng liên quan
    metadata : dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Kết quả sinh câu trả lời — chung cho mọi generator (Ollama/Gemini/Claude)."""
    answer       : str
    tokens_used  : int
    finish_reason: str
    sources      : list[dict]   # [{"filename": ..., "page": ..., "doc_id": ...}]
