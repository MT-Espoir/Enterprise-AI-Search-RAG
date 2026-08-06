import logging
from google import genai
from google.genai import types

from ..schemas import RetrievedChunk, GenerationResult
from .base_generator import BaseGenerator
from .prompts import GENERATOR_SYSTEM_INSTRUCTION_GEMINI

logger = logging.getLogger(__name__)

GEMINI_MODEL_DEFAULT = "gemini-3.1-flash-lite"


class GeminiGenerator(BaseGenerator):
    """
    Sinh câu trả lời bằng Google Gemini.
    Phần ghép context / trích nguồn / dựng prompt dùng chung ở BaseGenerator.
    """

    SYSTEM_INSTRUCTION = GENERATOR_SYSTEM_INSTRUCTION_GEMINI

    def __init__(self, api_key: str, model_name: str = None):
        self.client     = genai.Client(api_key=api_key)
        self.model_name = model_name or GEMINI_MODEL_DEFAULT

    def generate(self, question: str, chunks: list[RetrievedChunk], history: list[dict] = None, max_retries: int = 3) -> GenerationResult:
        """
        Sinh câu trả lời từ câu hỏi và danh sách chunks đã rerank. Có tự động retry nếu dính Rate Limit 429.
        Hỗ trợ lịch sử trò chuyện (history).
        """
        import time
        user_prompt, sources = self._prepare(question, chunks)

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
