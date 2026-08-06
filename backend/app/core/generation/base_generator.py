"""Lớp trừu tượng cho MỌI nhà cung cấp sinh câu trả lời (generation).

Cùng vai trò với `BaseRetriever`/`BaseParser`/`BaseEmbedder` ở các tầng khác —
trước đây riêng tầng generation không có lớp chung, mỗi generator tự do định
nghĩa nên phát sinh trùng lặp code VÀ lệch hành vi giữa các nhà cung cấp.

Phân chia trách nhiệm:
  - `generate()` là phần DUY NHẤT bắt buộc mỗi nhà cung cấp tự cài đặt (gọi API
    riêng, đọc token/finish_reason theo định dạng riêng).
  - Phần chuẩn bị dữ liệu (ghép context, trích nguồn, dựng prompt người dùng)
    nằm ở đây và DÙNG CHUNG — nhờ vậy đổi nhà cung cấp không làm đổi cách trình
    bày context hay danh sách trích dẫn, giúp so sánh chất lượng giữa các nhà
    cung cấp trở nên công bằng khi benchmark.

Thêm một nhà cung cấp mới = tạo 1 file kế thừa lớp này + cài `generate()` +
đăng ký trong `generator_factory.py`.
"""
from abc import ABC, abstractmethod

from ..schemas import GenerationResult, RetrievedChunk


class BaseGenerator(ABC):
    """Interface chung cho generator. Mọi nhà cung cấp PHẢI trả về GenerationResult."""

    # Mỗi nhà cung cấp tự đặt (prompt hệ thống có thể khác nhau theo model).
    SYSTEM_INSTRUCTION: str = ""

    # ── Phần DÙNG CHUNG (không nên override trừ khi có lý do rõ ràng) ──────────

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Ghép các chunk thành khối context cho prompt.

        Định dạng THỐNG NHẤT cho mọi nhà cung cấp (trước đây Gemini dùng định dạng
        riêng `[1] (Nguồn: ...)` còn Local/Claude dùng `--- Tài liệu 1 | ... ---`,
        khiến cùng một câu hỏi lại đưa context khác nhau tùy nhà cung cấp — không
        so sánh được khi benchmark).
        """
        if not chunks:
            return "Không có tài liệu tham khảo nào."

        parts = []
        for i, c in enumerate(chunks, 1):
            page_info = f" (Trang {c.page})" if c.page else ""
            parts.append(f"--- Tài liệu {i} | Nguồn: {c.filename}{page_info} ---\n{c.text}\n")
        return "\n".join(parts)

    def _extract_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        """Trích danh sách nguồn để trả kèm câu trả lời, CÓ KHỬ TRÙNG LẶP theo
        (doc_id, page).

        Trước đây chỉ Gemini khử trùng lặp, còn Local/Claude thì không — nhiều
        chunk cùng một trang sẽ hiện trích dẫn lặp lại cho người đọc. Nay thống
        nhất khử trùng lặp cho mọi nhà cung cấp.
        """
        seen = set()
        sources = []
        for c in chunks:
            key = (c.doc_id, c.page)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"doc_id": c.doc_id, "filename": c.filename, "page": c.page})
        return sources

    def _build_user_prompt(self, question: str, context: str) -> str:
        """Dựng phần prompt của người dùng — giống hệt nhau ở cả 3 generator cũ."""
        return (
            f"[TÀI LIỆU THAM KHẢO]\n\n{context}\n\n"
            f"---\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Câu trả lời:"
        )

    def _prepare(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
        """Gộp 3 bước chuẩn bị thường dùng ở đầu mỗi generate(): trả (user_prompt, sources)."""
        return self._build_user_prompt(question, self._format_context(chunks)), self._extract_sources(chunks)

    # ── Phần mỗi nhà cung cấp PHẢI tự cài đặt ─────────────────────────────────

    @abstractmethod
    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict] = None,
    ) -> GenerationResult:
        """Gọi LLM của nhà cung cấp và trả về GenerationResult.

        `history` là lịch sử hội thoại dạng [{"role": "user"|"assistant", "content": str}].
        """
        ...
