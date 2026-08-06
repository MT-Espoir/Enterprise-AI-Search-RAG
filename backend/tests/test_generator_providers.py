"""
test_generator_providers.py — Test tầng generation sau khi thêm BaseGenerator +
factory đa nhà cung cấp. KHÔNG gọi API thật (không tốn tiền, không cần key thật):
chỉ kiểm tra phần dùng chung + việc chọn nhà cung cấp theo config.

Chạy: python backend/tests/test_generator_providers.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.generation import (
    BaseGenerator, build_generator, VALID_PROVIDERS,
    GeminiGenerator, LocalGenerator, ClaudeGenerator, OpenAIGenerator,
)
from app.core.schemas import RetrievedChunk

FAKE = {"GOOGLE_API_KEY": "k", "ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k",
        "OLLAMA_BASE_URL": "http://localhost:11434", "LOCAL_LLM_MODEL": "qwen2.5:3b-instruct"}


def _chunks():
    # 2 chunk ĐẦU cùng (doc_id, page) → phải bị gộp khi trích nguồn
    return [
        RetrievedChunk("c1", "d1", "a.pdf", 3, "nội dung 1", 0.9),
        RetrievedChunk("c2", "d1", "a.pdf", 3, "nội dung 2", 0.8),
        RetrievedChunk("c3", "d1", "a.pdf", 7, "nội dung 3", 0.7),
    ]


def test_factory_chon_dung_nha_cung_cap():
    cases = [("local", LocalGenerator), ("gemini", GeminiGenerator),
             ("claude", ClaudeGenerator), ("openai", OpenAIGenerator)]
    for provider, cls in cases:
        gen = build_generator({**FAKE, "LLM_PROVIDER": provider})
        assert isinstance(gen, cls), f"{provider} -> {type(gen)}"
        assert isinstance(gen, BaseGenerator), f"{provider} không kế thừa BaseGenerator"
    print("✅ factory chọn đúng 4 nhà cung cấp, tất cả đều là BaseGenerator")


def test_provider_khong_hop_le_fallback_local():
    gen = build_generator({**FAKE, "LLM_PROVIDER": "khong-ton-tai"})
    assert isinstance(gen, LocalGenerator)
    # không set gì cũng phải ra local
    assert isinstance(build_generator({**FAKE}), LocalGenerator)
    print("✅ provider sai/thiếu → fallback an toàn về local")


def test_gemini_model_doc_tu_config():
    """Bug cũ: chat.py không truyền model nên config GEMINI_MODEL bị bỏ qua."""
    gen = build_generator({**FAKE, "LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-1.5-pro"})
    assert gen.model_name == "gemini-1.5-pro", gen.model_name
    # không set → dùng default trong code
    gen2 = build_generator({**FAKE, "LLM_PROVIDER": "gemini"})
    assert gen2.model_name == "gemini-3.1-flash-lite", gen2.model_name
    print("✅ GEMINI_MODEL từ config được tôn trọng (bug cũ đã hết)")


def test_khu_trung_lap_nguon_tren_moi_provider():
    """Trước đây chỉ Gemini khử trùng lặp; local/claude trả nguồn lặp lại."""
    chunks = _chunks()
    for provider in ("local", "gemini", "claude", "openai"):
        gen = build_generator({**FAKE, "LLM_PROVIDER": provider})
        sources = gen._extract_sources(chunks)
        assert len(sources) == 2, f"{provider}: {len(sources)} nguồn (mong đợi 2)"
        assert sources[0]["page"] == 3 and sources[1]["page"] == 7
    print("✅ mọi provider đều khử trùng lặp nguồn theo (doc_id, page)")


def test_context_thong_nhat_giua_cac_provider():
    chunks = _chunks()
    outs = {p: build_generator({**FAKE, "LLM_PROVIDER": p})._format_context(chunks)
            for p in ("local", "gemini", "claude", "openai")}
    assert len(set(outs.values())) == 1, "định dạng context KHÔNG thống nhất giữa các provider"
    # chunk rỗng không được crash
    assert "Không có tài liệu" in build_generator({**FAKE})._format_context([])
    print("✅ định dạng context thống nhất, xử lý chunk rỗng an toàn")


def test_openai_base_url_doi_duoc_nha_cung_cap():
    """Cùng 1 class phục vụ ChatGPT/DeepSeek/GLM — chỉ khác base_url."""
    gen = build_generator({**FAKE, "LLM_PROVIDER": "openai",
                           "OPENAI_BASE_URL": "https://api.deepseek.com",
                           "OPENAI_MODEL": "deepseek-chat"})
    assert gen.base_url == "https://api.deepseek.com" and gen.model_name == "deepseek-chat"
    print("✅ đổi base_url → dùng được nhà cung cấp khác cùng giao thức")


def test_base_generator_khong_the_khoi_tao_truc_tiep():
    try:
        BaseGenerator()
        raise AssertionError("BaseGenerator lẽ ra phải là abstract")
    except TypeError:
        print("✅ BaseGenerator là abstract (bắt buộc cài đặt generate())")


if __name__ == "__main__":
    test_factory_chon_dung_nha_cung_cap()
    test_provider_khong_hop_le_fallback_local()
    test_gemini_model_doc_tu_config()
    test_khu_trung_lap_nguon_tren_moi_provider()
    test_context_thong_nhat_giua_cac_provider()
    test_openai_base_url_doi_duoc_nha_cung_cap()
    test_base_generator_khong_the_khoi_tao_truc_tiep()
    print("\n🎉 TẤT CẢ TEST GENERATION PASS")
