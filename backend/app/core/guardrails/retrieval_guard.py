import logging
from dataclasses import dataclass, replace

from ..retriever import RetrievedChunk
from .patterns import _PROMPT_INJECTION_PATTERN, _JAILBREAK_PATTERN

logger = logging.getLogger(__name__)

_REDACTION_MARKER = "[NỘI DUNG ĐÃ BỊ LỌC — NGHI VẤN CHỨA CHỈ THỊ ẨN]"


@dataclass
class SanitizeReport:
    chunks_sanitized: int = 0
    substrings_redacted: int = 0


def _redact(text: str) -> tuple[str, int]:
    count = 0

    def _sub(_match) -> str:
        nonlocal count
        count += 1
        return _REDACTION_MARKER

    new_text = _PROMPT_INJECTION_PATTERN.sub(_sub, text)
    new_text = _JAILBREAK_PATTERN.sub(_sub, new_text)
    return new_text, count


def sanitize_chunks(chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], SanitizeReport]:
    """Retrieval Guardrail (Tier 1, Layer 4) — Document Sanitizer.

    Quét injection/jailbreak pattern (dùng lại patterns.py) trên chunk.text —
    chỉ redact ĐÚNG đoạn khớp, KHÔNG xóa cả chunk, để giữ nội dung pháp lý hợp
    pháp còn lại. Tạo RetrievedChunk MỚI qua dataclasses.replace() thay vì
    mutate in-place (tránh side-effect aliasing với candidates/top_chunks dùng
    chung tham chiếu). Fail-open theo từng chunk: 1 chunk lỗi không làm rớt cả
    danh sách."""
    report = SanitizeReport()
    sanitized_chunks: list[RetrievedChunk] = []

    for chunk in chunks:
        try:
            new_text, count = _redact(chunk.text)
            if count > 0:
                new_metadata = {**chunk.metadata, "sanitized": True}
                sanitized_chunks.append(replace(chunk, text=new_text, metadata=new_metadata))
                report.chunks_sanitized += 1
                report.substrings_redacted += count
            else:
                sanitized_chunks.append(chunk)
        except Exception as exc:
            logger.warning(
                f"Retrieval guardrail lỗi ở chunk {getattr(chunk, 'chunk_id', '?')} ({exc}) — "
                f"giữ nguyên chunk gốc."
            )
            sanitized_chunks.append(chunk)

    return sanitized_chunks, report
