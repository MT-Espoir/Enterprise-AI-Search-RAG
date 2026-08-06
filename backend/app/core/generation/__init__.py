"""Nhóm SINH CÂU TRẢ LỜI (generation): từ câu hỏi + chunk đã chọn → câu trả lời có trích dẫn.

Mọi nhà cung cấp kế thừa `BaseGenerator` (mỗi nhà cung cấp 1 file) và trả về cùng
`GenerationResult`, nên hoán đổi được cho nhau. Chọn qua config LLM_PROVIDER:

    local  → Ollama chạy máy (mặc định)
    gemini → Google Gemini
    claude → Anthropic Claude
    openai → mọi endpoint chuẩn OpenAI (ChatGPT, DeepSeek, GLM, Groq...)

Dùng `build_generator(config)` thay vì khởi tạo trực tiếp — xem generator_factory.py.
"""
from .base_generator import BaseGenerator
from .generator_factory import build_generator, VALID_PROVIDERS
from .gemini_generator import GeminiGenerator
from .local_generator import LocalGenerator
from .claude_generator import ClaudeGenerator
from .openai_generator import OpenAIGenerator

__all__ = [
    "BaseGenerator", "build_generator", "VALID_PROVIDERS",
    "GeminiGenerator", "LocalGenerator", "ClaudeGenerator", "OpenAIGenerator",
]
