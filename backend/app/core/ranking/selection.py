"""Chọn top-k SAU rerank — tách khỏi bước chấm điểm (multi_doc_selection_plan.md, H1).

Vì sao: với câu so sánh / tổng hợp nhiều tài liệu, top-k theo điểm cross-encoder thường dồn vào MỘT tài liệu
(chunk của tài liệu đó cùng khớp câu hỏi gốc) → Generator chỉ thấy một nửa thông tin. Đo offline 2026-10-25 trên
17 câu nhóm multi_doc: Decomposition đưa đủ tài liệu vào ứng viên ở 82% câu nhưng top-3 theo điểm chỉ còn 29%;
chọn đa dạng tài liệu (select_diverse) trên CÙNG ứng viên: 29% → 76% (áp mọi câu) / 59% (chỉ câu so sánh).

Chỉ dùng cho câu SO SÁNH: áp cho mọi câu làm 8/194 câu đơn tài liệu mất chunk chứa đáp án (đáp án ở chunk thứ 2–3
của CÙNG tài liệu bị thay bằng tài liệu không liên quan) — xem is_comparison_query() ở query_processor.py.
"""
import hashlib
import re

from ..schemas import RetrievedChunk


def _content_key(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text or "").strip().lower().encode("utf-8")).hexdigest()


def dedupe_identical(ranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Bỏ chunk trùng NGUYÊN VĂN, giữ bản đứng trước (điểm cao hơn) — bản trùng của cùng một văn bản (vd
    45.signed.pdf ≡ Boluatlaodong2019.pdf, 309/309 chunk) không được chiếm 2 suất "đa dạng"."""
    out, seen = [], set()
    for chunk in ranked:
        key = _content_key(chunk.text)
        if key not in seen:
            seen.add(key)
            out.append(chunk)
    return out


def select_diverse(ranked: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
    """ranked: đã sắp theo điểm rerank giảm dần. Mỗi tài liệu (doc_id) 1 chunk điểm cao nhất trước, đủ k tài liệu
    rồi thôi; ít tài liệu hơn k → lấp bằng chunk điểm cao còn lại. Không ngưỡng điểm: khoảng cách điểm giữa tài liệu
    ĐÚNG với top-1 có trung vị chỉ 0.04 (thang Sigmoid 0–1) → ngưỡng loại nhầm tài liệu cần lấy."""
    ranked = dedupe_identical(ranked)
    picked, docs = [], set()
    for chunk in ranked:
        if chunk.doc_id not in docs:
            picked.append(chunk)
            docs.add(chunk.doc_id)
            if len(picked) == k:
                return picked
    ids = {c.chunk_id for c in picked}
    for chunk in ranked:
        if len(picked) == k:
            break
        if chunk.chunk_id not in ids:
            picked.append(chunk)
            ids.add(chunk.chunk_id)
    return picked
