import logging
from dataclasses import dataclass
from google import genai
from google.genai import types
from .retriever import RetrievedChunk
from .prompts import GENERATOR_SYSTEM_INSTRUCTION_GEMINI

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    answer       : str
    tokens_used  : int
    finish_reason: str
    sources      : list[dict]   # [{"filename": ..., "page": ..., "doc_id": ...}]


class Generator:
    """
    Bước cuối trong RAG pipeline: Sinh câu trả lời bằng Google Gemini.
    Nhận câu hỏi + top chunks → build prompt → gọi LLM → trả về câu trả lời.
    """

    SYSTEM_INSTRUCTION = GENERATOR_SYSTEM_INSTRUCTION_GEMINI

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite"):
        self.client     = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Định dạng danh sách chunks thành chuỗi context cho prompt."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source_info = f"Nguồn: {chunk.filename}"
            if chunk.page:
                source_info += f", Trang {chunk.page}"
            parts.append(f"[{i}] ({source_info})\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        """Trích xuất thông tin nguồn từ các chunks để trả về cùng câu trả lời."""
        seen = set()
        sources = []
        for chunk in chunks:
            key = (chunk.doc_id, chunk.page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "doc_id":   chunk.doc_id,
                    "filename": chunk.filename,
                    "page":     chunk.page,
                })
        return sources

    def generate(self, question: str, chunks: list[RetrievedChunk], history: list[dict] = None, max_retries: int = 3) -> GenerationResult:
        """
        Sinh câu trả lời từ câu hỏi và danh sách chunks đã rerank. Có tự động retry nếu dính Rate Limit 429.
        Hỗ trợ lịch sử trò chuyện (history).
        """
        import time
        context = self._format_context(chunks)
        sources = self._extract_sources(chunks)

        user_prompt = (
            f"[TÀI LIỆU THAM KHẢO]\n\n{context}\n\n"
            f"---\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Câu trả lời:"
        )

        # Xây dựng danh sách contents (history + current turn)
        contents = []
        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.SYSTEM_INSTRUCTION,
                        temperature=0.2,          
                        max_output_tokens=2048,
                    ),
                )
                
                candidate = response.candidates[0]
                answer = candidate.content.parts[0].text

                usage = response.usage_metadata
                tokens_used = (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)
                finish_reason = str(candidate.finish_reason).split(".")[-1]

                logger.info(f"Generated answer | tokens={tokens_used} | reason={finish_reason}")

                return GenerationResult(
                    answer=answer,
                    tokens_used=tokens_used,
                    finish_reason=finish_reason,
                    sources=sources,
                )
            except Exception as e:
                # Nếu gặp lỗi Rate Limit (429) hoặc lỗi mạng
                if attempt == max_retries - 1:
                    logger.error(f"Generator thất bại sau {max_retries} lần thử: {e}")
                    raise
                wait_time = 20 * (attempt + 1)
                logger.warning(f"Lỗi Generator (có thể do Rate Limit). Thử lại sau {wait_time}s... (Lỗi: {e})")
                time.sleep(wait_time)
