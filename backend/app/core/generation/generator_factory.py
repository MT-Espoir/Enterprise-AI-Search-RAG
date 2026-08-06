"""Factory chọn nhà cung cấp sinh câu trả lời theo config — cùng mẫu với
`retriever_factory.build_retriever()` ở tầng retrieval.

Vì sao cần: trước đây `chat.py` tự viết chuỗi if/elif, nên THÊM một nhà cung cấp
là phải sửa tầng API. Nay `chat.py` chỉ gọi `build_generator(config)`; muốn thêm
nhà cung cấp chỉ cần khai báo thêm một nhánh Ở ĐÂY.
"""
import logging

from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)

# Các giá trị hợp lệ của LLM_PROVIDER
VALID_PROVIDERS = ("local", "gemini", "claude", "openai")


def build_generator(config) -> BaseGenerator:
    """Dựng generator theo `config["LLM_PROVIDER"]`.

    `config` là mapping bất kỳ (Flask app.config hoặc dict) — nhờ vậy benchmark/
    script dùng lại được mà không cần Flask app context.

    Import từng SDK BÊN TRONG nhánh tương ứng (lazy): máy chỉ cài SDK của nhà cung
    cấp đang dùng vẫn chạy được, không bắt buộc cài đủ google-genai + anthropic.
    """
    provider = (config.get("LLM_PROVIDER") or "local").strip().lower()

    if provider == "claude":
        from .claude_generator import ClaudeGenerator
        return ClaudeGenerator(
            api_key=config.get("ANTHROPIC_API_KEY"),
            model_name=config.get("ANTHROPIC_MODEL"),
        )

    if provider == "gemini":
        from .gemini_generator import GeminiGenerator
        # FIX: trước đây chat.py gọi Generator(api_key=...) mà KHÔNG truyền model,
        # nên config GEMINI_MODEL bị bỏ qua hoàn toàn và luôn chạy default cứng
        # trong code (cùng loại bug từng gặp với LOCAL_LLM_MODEL).
        return GeminiGenerator(
            api_key=config.get("GOOGLE_API_KEY"),
            model_name=config.get("GEMINI_MODEL"),
        )

    if provider == "openai":
        from .openai_generator import OpenAIGenerator
        return OpenAIGenerator(
            api_key=config.get("OPENAI_API_KEY"),
            model_name=config.get("OPENAI_MODEL"),
            base_url=config.get("OPENAI_BASE_URL"),
        )

    if provider not in VALID_PROVIDERS:
        logger.warning(
            f"LLM_PROVIDER={provider!r} không hợp lệ (hợp lệ: {VALID_PROVIDERS}) "
            f"— fallback về 'local'."
        )

    from .local_generator import LocalGenerator
    return LocalGenerator(
        base_url=config.get("OLLAMA_BASE_URL"),
        model_name=config.get("LOCAL_LLM_MODEL"),
    )
