import logging
from dataclasses import dataclass, field

from .patterns import (
    _PROMPT_INJECTION_PATTERN,
    _JAILBREAK_PATTERN,
    _CCCD_CMND_PATTERN,
    _PHONE_PATTERN,
    _EMAIL_PATTERN,
)

logger = logging.getLogger(__name__)

# Tránh chặn câu tỉnh lược cực ngắn hợp lệ (vd "Vì sao?") / prompt-stuffing câu
# hỏi siêu dài.
DEFAULT_MIN_LENGTH = 2
DEFAULT_MAX_LENGTH = 2000


@dataclass
class InputCheckResult:
    allowed: bool
    reason: str | None = None  # "too_short" | "too_long" | "prompt_injection" | "jailbreak"
    pii_detected: list[str] = field(default_factory=list)  # flag-only, KHÔNG ảnh hưởng allowed


def check_input(
    question: str,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> InputCheckResult:
    """Input Guardrail (Tier 1, Layer 1) — regex/heuristic, fail-open tuyệt đối.

    Thứ tự: length -> prompt injection -> jailbreak (dừng ở lỗi đầu tiên gặp)
    -> PII (luôn chạy, không ảnh hưởng allowed — xem patterns.py)."""
    try:
        if not isinstance(question, str):
            return InputCheckResult(allowed=False, reason="too_short")

        stripped = question.strip()
        if len(stripped) < min_length:
            return InputCheckResult(allowed=False, reason="too_short")
        if len(stripped) > max_length:
            return InputCheckResult(allowed=False, reason="too_long")

        if _PROMPT_INJECTION_PATTERN.search(stripped):
            return InputCheckResult(allowed=False, reason="prompt_injection")
        if _JAILBREAK_PATTERN.search(stripped):
            return InputCheckResult(allowed=False, reason="jailbreak")

        pii_detected: list[str] = []
        if _CCCD_CMND_PATTERN.search(stripped):
            pii_detected.append("cccd_cmnd")
        if _PHONE_PATTERN.search(stripped):
            pii_detected.append("phone")
        if _EMAIL_PATTERN.search(stripped):
            pii_detected.append("email")

        return InputCheckResult(allowed=True, pii_detected=pii_detected)
    except Exception as exc:
        logger.warning(f"Input guardrail lỗi nội bộ ({exc}) — fail-open, cho phép qua.")
        return InputCheckResult(allowed=True)
