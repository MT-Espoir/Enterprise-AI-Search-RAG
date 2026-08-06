"""Generator cho MỌI dịch vụ dùng chuẩn API OpenAI (Chat Completions).

Một file phủ nhiều nhà cung cấp vì họ chung GIAO THỨC, chỉ khác endpoint:
  - OpenAI (ChatGPT) : https://api.openai.com/v1
  - DeepSeek         : https://api.deepseek.com
  - GLM (Zhipu)      : endpoint tương thích OpenAI của Zhipu
  - Groq / Together / OpenRouter / vLLM tự host ...

Tách 3 file riêng cho ChatGPT/DeepSeek/GLM sẽ là 3 bản sao gần như y hệt — đúng
kiểu trùng lặp mà BaseGenerator vừa được tạo ra để loại bỏ. Muốn dùng nhà cung
cấp nào chỉ cần đổi OPENAI_BASE_URL + OPENAI_MODEL trong .env, không cần sửa code.

Gọi qua HTTP thuần bằng `requests` (đã có sẵn trong dự án) thay vì thêm SDK
`openai` — endpoint /chat/completions đủ đơn giản và tránh thêm phụ thuộc mới.
"""
import logging
from typing import List

import requests

from ..schemas import GenerationResult, RetrievedChunk
from .base_generator import BaseGenerator
from .prompts import GENERATOR_SYSTEM_INSTRUCTION_LOCAL

logger = logging.getLogger(__name__)

OPENAI_BASE_URL_DEFAULT = "https://api.openai.com/v1"
OPENAI_MODEL_DEFAULT = "gpt-4o-mini"

# Endpoint dựng sẵn cho vài nhà cung cấp phổ biến — tiện tra cứu khi cấu hình .env
KNOWN_BASE_URLS = {
    "openai":   "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "glm":      "https://open.bigmodel.cn/api/paas/v4",
}


class OpenAIGenerator(BaseGenerator):
    """Sinh câu trả lời qua endpoint tương thích chuẩn OpenAI Chat Completions."""

    SYSTEM_INSTRUCTION = GENERATOR_SYSTEM_INSTRUCTION_LOCAL

    def __init__(self, api_key: str, model_name: str = None, base_url: str = None,
                 max_tokens: int = 2048, timeout: int = 120):
        if not api_key:
            raise ValueError("OpenAIGenerator cần API key (OPENAI_API_KEY trong .env).")
        self.api_key = api_key
        self.model_name = model_name or OPENAI_MODEL_DEFAULT
        self.base_url = (base_url or OPENAI_BASE_URL_DEFAULT).rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout
        logger.info(f"Khởi tạo OpenAIGenerator (model={self.model_name}, base_url={self.base_url})")

    def generate(self, question: str, chunks: List[RetrievedChunk], history: list[dict] = None) -> GenerationResult:
        user_prompt, sources = self._prepare(question, chunks)

        messages = [{"role": "system", "content": self.SYSTEM_INSTRUCTION}]
        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": user_prompt})

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.1,   # thấp cho RAG (bám context, ít bịa)
                    "max_tokens": self.max_tokens,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as exc:
            # Trả rõ nội dung lỗi từ nhà cung cấp (sai model/hết quota/sai key...)
            body = exc.response.text[:300] if exc.response is not None else ""
            logger.error(f"Lỗi HTTP từ {self.base_url}: {exc} | {body}")
            raise
        except Exception as exc:
            logger.error(f"Lỗi khi gọi {self.base_url}: {exc}")
            raise

        choice = (data.get("choices") or [{}])[0]
        answer = (choice.get("message") or {}).get("content", "") or ""
        finish_reason = choice.get("finish_reason", "stop")
        usage = data.get("usage") or {}
        tokens_used = (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)

        logger.info(f"OpenAI-compatible Generated | model={self.model_name} | "
                    f"tokens={tokens_used} | reason={finish_reason}")

        return GenerationResult(
            answer=answer.strip(),
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            sources=sources,
        )
