import logging
from typing import List

import anthropic

from .generator import GenerationResult, RetrievedChunk
from .prompts import GENERATOR_SYSTEM_INSTRUCTION_LOCAL

logger = logging.getLogger(__name__)

# Mặc định Haiku 4.5 (rẻ nhất, $1/$5 mỗi triệu token) — đủ tốt cho RAG generation
# và làm judge benchmark chạy nhiều lần. Đổi qua env ANTHROPIC_MODEL nếu cần model
# mạnh hơn (claude-sonnet-5 / claude-opus-5). KHÔNG gắn hậu tố ngày vào model id.
CLAUDE_MODEL_DEFAULT = "claude-haiku-4-5"


class ClaudeGenerator:
    """
    Sinh câu trả lời bằng Claude (Anthropic API), qua SDK chính thức `anthropic`.
    Cùng interface (GenerationResult) với Generator (Gemini) và LocalGenerator (Ollama)
    nên là drop-in thay thế ở composition root (chat.py) — chọn qua LLM_PROVIDER=claude.

    Haiku 4.5 là model pre-4.6: KHÔNG truyền tham số `thinking` (bỏ trống = không suy
    luận mở rộng, nhanh + rẻ, đúng nhu cầu RAG trả lời bám context). temperature vẫn
    được chấp nhận trên model này (khác Opus 5 / Sonnet 5).
    """

    SYSTEM_INSTRUCTION = GENERATOR_SYSTEM_INSTRUCTION_LOCAL

    def __init__(self, api_key: str, model_name: str = CLAUDE_MODEL_DEFAULT, max_tokens: int = 2048):
        if not api_key:
            raise ValueError("ClaudeGenerator cần ANTHROPIC_API_KEY (chưa cấu hình trong .env).")
        # SDK tự retry 429/5xx/lỗi mạng với backoff — nâng max_retries lên 3 cho chắc.
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3)
        self.model_name = model_name
        self.max_tokens = max_tokens
        logger.info(f"Khởi tạo ClaudeGenerator (Anthropic API, model={self.model_name})")

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "Không có tài liệu tham khảo nào."
        parts = []
        for i, c in enumerate(chunks, 1):
            page_info = f" (Trang {c.page})" if c.page else ""
            parts.append(f"--- Tài liệu {i} | Nguồn: {c.filename}{page_info} ---\n{c.text}\n")
        return "\n".join(parts)

    def _extract_sources(self, chunks: List[RetrievedChunk]) -> List[dict]:
        return [{"doc_id": c.doc_id, "filename": c.filename, "page": c.page} for c in chunks]

    def generate(self, question: str, chunks: List[RetrievedChunk], history: list[dict] = None) -> GenerationResult:
        context = self._format_context(chunks)
        sources = self._extract_sources(chunks)

        user_prompt = (
            f"[TÀI LIỆU THAM KHẢO]\n\n{context}\n\n"
            f"---\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Câu trả lời:"
        )

        # Anthropic: system là tham số top-level RIÊNG (không nằm trong messages).
        messages = []
        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=0.1,  # thấp cho RAG (bám context, ít bịa)
                system=self.SYSTEM_INSTRUCTION,
                messages=messages,
            )
        except Exception as exc:
            logger.error(f"Lỗi khi gọi Claude (Anthropic API): {exc}")
            raise

        # response.content là list block; ghép các block type="text".
        answer = "".join(block.text for block in response.content if block.type == "text").strip()
        finish_reason = response.stop_reason  # "end_turn" | "max_tokens" | "refusal" | ...

        # refusal: classifier từ chối → answer rỗng. Trả về fallback rõ ràng thay vì chuỗi rỗng
        # (hiếm với Haiku cho câu hỏi pháp lý hợp lệ, nhưng vẫn phòng để không crash UI).
        if finish_reason == "refusal" and not answer:
            answer = "Tôi không thể trả lời câu hỏi này."

        tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
        logger.info(f"Claude Generated | tokens={tokens_used} | reason={finish_reason}")

        return GenerationResult(
            answer=answer,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            sources=sources,
        )
