"""Header ngữ cảnh tài liệu (contextual header) — một dòng nhận diện tài liệu
("[Báo cáo tài chính riêng năm 2016 của Tổng Công ty ... (Vinaconex), ...]") ghép
vào trước nội dung chunk khi EMBED / đánh chỉ mục BM25 / RERANK.

Vì sao: chunk là mảnh trang, phần lớn không chứa năm / loại báo cáo / tên văn bản
(benchmark 2026-10-22: chỉ 36% chunk BCTC chứa năm) → retriever không phân biệt
được các tài liệu rất giống nhau. PoC trên bản sao snapshot: rerank doc hit@3 nhóm
BCTC 0.577 → 1.000 (xem roadmap_tasklist/contextual_header_rollout_plan.md).

Quy ước dữ liệu (bất biến):
  - ChromaDB `documents` LUÔN là nội dung gốc — Generator, guardrails, trích dẫn,
    UI không thấy header trong nội dung đoạn.
  - `metadata["context_header"]` CHỈ được ghi khi embedding của chunk đó thật sự
    được tính từ build_context_text(header, text) → nhìn metadata là biết vector
    đã có header hay chưa.

Killswitch CONTEXT_HEADER_ENABLED (config.py, mặc định tắt): tắt → ingest không ghép
header, BM25/Reranker/Generator bỏ qua metadata["context_header"]. KHÔNG gỡ được
header đã "nướng" vào embedding — muốn gỡ hẳn phải embed lại không header.
"""

_enabled = False


def set_enabled(enabled: bool) -> None:
    """Gọi 1 lần trong create_app() theo config; benchmark/test tự gọi khi cần."""
    global _enabled
    _enabled = bool(enabled)


def is_enabled() -> bool:
    return _enabled


def build_context_text(header: str | None, text: str) -> str:
    """Công thức DUY NHẤT ghép header + nội dung — dùng chung cho embedding lúc ingest,
    embedding khi embed lại, BM25 và Reranker để mọi bề mặt nhìn cùng một chuỗi."""
    header = (header or "").strip()
    return f"{header}\n{text}" if header else text


def header_of(metadata: dict | None) -> str:
    """Header của chunk nếu killswitch bật, ngược lại "" (bề mặt query-time bỏ qua)."""
    if not _enabled:
        return ""
    return ((metadata or {}).get("context_header") or "").strip()


def context_text_for(text: str, metadata: dict | None) -> str:
    """Chuỗi BM25/Reranker dùng cho 1 chunk — nội dung gốc khi killswitch tắt."""
    return build_context_text(header_of(metadata), text)
