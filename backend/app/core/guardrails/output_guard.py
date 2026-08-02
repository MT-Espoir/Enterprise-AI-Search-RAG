import logging
from dataclasses import dataclass, field

from ..retriever import RetrievedChunk
from .patterns import _CITATION_PATTERN

logger = logging.getLogger(__name__)


@dataclass
class OutputCheckResult:
    warning: bool = False
    reason: str | None = None  # "citation_mismatch" | None
    unverified_citations: list[str] = field(default_factory=list)


def check_output(answer: str, top_chunks: list[RetrievedChunk]) -> OutputCheckResult:
    """Output Guardrail (Tier 1, Layer 7) — citation cross-check.

    Trích "Điều <số>" từ answer, kiểm tra substring (case-insensitive) có xuất
    hiện trong text của top_chunks (chunk THẬT SỰ đưa vào LLM, sau
    sanitize/rerank) hay không. CHỈ CẢNH BÁO, không chặn."""
    try:
        mentioned = {m.group().strip() for m in _CITATION_PATTERN.finditer(answer or "")}
        if not mentioned:
            return OutputCheckResult()

        context_text = "\n".join(c.text for c in top_chunks).lower()
        unverified = [c for c in mentioned if c.lower() not in context_text]

        if unverified:
            return OutputCheckResult(
                warning=True,
                reason="citation_mismatch",
                unverified_citations=sorted(unverified),
            )
        return OutputCheckResult()
    except Exception as exc:
        logger.warning(f"Output guardrail lỗi nội bộ ({exc}) — fail-open, không gắn cảnh báo.")
        return OutputCheckResult()
