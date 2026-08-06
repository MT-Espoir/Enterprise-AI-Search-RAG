import logging
import json
import requests
from typing import List

from ..schemas import GenerationResult, RetrievedChunk
from .base_generator import BaseGenerator
from .prompts import GENERATOR_SYSTEM_INSTRUCTION_LOCAL

logger = logging.getLogger(__name__)

class LocalGenerator(BaseGenerator):
    """
    Sử dụng Ollama (Local LLM) để sinh câu trả lời.
    Chạy hoàn toàn offline, tốc độ phụ thuộc vào cấu hình máy tính (CPU/GPU).
    """

    SYSTEM_INSTRUCTION = GENERATOR_SYSTEM_INSTRUCTION_LOCAL

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "qwen2.5:3b-instruct"):
        """
        Args:
            base_url: URL của Ollama server (mặc định là localhost:11434)
            model_name: Tên model đã tải trong Ollama (VD: llama3.2:1b, qwen2.5:0.5b)
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        logger.info(f"Khởi tạo Local Generator với Ollama (Model: {self.model_name}) tại {self.base_url}")

    def generate(self, question: str, chunks: List[RetrievedChunk], history: list[dict] = None, max_retries: int = 1) -> GenerationResult:
        """
        Sinh câu trả lời từ câu hỏi và danh sách chunks đã rerank thông qua Ollama.
        """
        user_prompt, sources = self._prepare(question, chunks)

        # Xây dựng Messages (Chat format)
        messages = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION}
        ]

        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("content", "")})
        
        messages.append({"role": "user", "content": user_prompt})

        # Gọi API Ollama
        api_endpoint = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Nhiệt độ thấp cho RAG
                "num_predict": 1024  # Max tokens
            }
        }

        try:
            logger.info(f"Đang gửi request tới Ollama ({self.model_name})... (Sẽ mất thời gian nếu chạy CPU)")
            response = requests.post(api_endpoint, json=payload, timeout=300) # Đợi tối đa 5 phút cho CPU yếu
            response.raise_for_status()
            
            data = response.json()
            answer = data.get("message", {}).get("content", "")
            
            # Đếm token xấp xỉ theo số từ nếu Ollama không trả về chính xác
            tokens_used = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            finish_reason = data.get("done_reason", "stop")
            
            logger.info(f"Ollama Generated | tokens={tokens_used} | reason={finish_reason}")
            
            return GenerationResult(
                answer=answer,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                sources=sources,
            )

        except requests.exceptions.ConnectionError:
            error_msg = f"Không thể kết nối đến Ollama tại {self.base_url}. Vui lòng kiểm tra xem Ollama đã chạy chưa!"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
        except Exception as e:
            logger.error(f"Lỗi khi gọi Ollama: {e}")
            raise
