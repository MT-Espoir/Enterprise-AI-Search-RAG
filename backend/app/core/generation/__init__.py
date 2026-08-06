"""Nhóm SINH CÂU TRẢ LỜI (generation): từ câu hỏi + chunk đã chọn → câu trả lời có trích dẫn.

Ba nhà cung cấp cùng interface (trả GenerationResult), chọn qua config LLM_PROVIDER:
Ollama local (mặc định), Gemini, Claude.
"""
from .generator import Generator
from .local_generator import LocalGenerator
from .claude_generator import ClaudeGenerator

__all__ = ["Generator", "LocalGenerator", "ClaudeGenerator"]
