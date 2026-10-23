def calculate_hit_rate(retrieved_docs, expected_docs):
    """
    Hit Rate là 1 nếu ít nhất một tài liệu được lấy ra nằm trong danh sách mong đợi.
    Ngược lại là 0.
    """
    if not expected_docs:
        # Nếu câu hỏi không mong đợi tài liệu nào (negative question)
        return 1 if not retrieved_docs else 0

    for doc in retrieved_docs:
        if _is_doc_match(doc, expected_docs):
            return 1
    return 0


def calculate_mrr(retrieved_docs, expected_docs):
    """
    Mean Reciprocal Rank: 1/rank của tài liệu đúng đầu tiên.
    """
    if not expected_docs:
        return 1.0 if not retrieved_docs else 0.0

    for idx, doc in enumerate(retrieved_docs):
        if _is_doc_match(doc, expected_docs):
            return 1.0 / (idx + 1)
    return 0.0


def calculate_recall_at_k(retrieved_docs, expected_docs, k):
    """
    Recall = Số tài liệu đúng tìm được trong top K / Tổng số tài liệu đúng (Ground Truth)
    """
    if not expected_docs:
        return 1.0 if not retrieved_docs[:k] else 0.0

    retrieved_top_k = retrieved_docs[:k]
    matched_count = 0
    # Tính số match dựa trên expected docs (vì chunk có thể nhiều phần thuộc 1 expected doc)
    # Để tránh đếm lặp, ta sẽ tính xem bao nhiêu expected doc được cover
    covered_expected = set()
    for doc in retrieved_top_k:
        matched_idx = _find_match_idx(doc, expected_docs)
        if matched_idx is not None:
            covered_expected.add(matched_idx)

    return len(covered_expected) / len(expected_docs)


def calculate_precision_at_k(retrieved_docs, expected_docs, k):
    """
    Precision = Số tài liệu đúng tìm được trong top K / K
    """
    if not expected_docs:
        return 1.0 if not retrieved_docs[:k] else 0.0

    retrieved_top_k = retrieved_docs[:k]
    if not retrieved_top_k:
        return 0.0

    matched_count = 0
    for doc in retrieved_top_k:
        if _is_doc_match(doc, expected_docs):
            matched_count += 1

    return matched_count / k


def _is_doc_match(retrieved_doc, expected_docs):
    """Kiểm tra xem retrieved_doc có khớp với bất kỳ expected_doc nào không."""
    return _find_match_idx(retrieved_doc, expected_docs) is not None


def _find_match_idx(retrieved_doc, expected_docs):
    """Trả về index của expected_doc khớp với retrieved_doc, hoặc None."""
    ret_filename = retrieved_doc.filename
    ret_page = retrieved_doc.page

    for idx, exp in enumerate(expected_docs):
        exp_filename = exp.get("filename")
        exp_page = exp.get("page_num")

        if exp_filename == ret_filename:
            # Nếu page là None hoặc bằng nhau
            if exp_page is None or exp_page == ret_page:
                return idx
    return None


# ── Benchmark tổng quát (run_benchmark_full.py) ────────────────────────────────
# Nhãn đúng ở 2 mức, đo trên CÙNG danh sách retrieved đã xếp hạng:
#   - chunk: relevant_chunk_ids — các chunk chứa đủ thông tin trả lời.
#   - doc  : relevant_doc_groups — mỗi nhóm là 1 "đơn vị" tài liệu đúng; các bản trùng
#            nội dung (vd 45.signed.pdf / Boluatlaodong2019.pdf) gom chung 1 nhóm để
#            Recall không đòi retriever phải lấy cả 2 bản sao.
# Không có câu negative ở đây: retriever luôn trả top-k nên metric retrieval của câu
# negative vô nghĩa (luôn 0 theo calculate_hit_rate ở trên) — đo riêng bằng refusal.


def chunk_level_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    top_k = retrieved_ids[:k]
    found = sum(1 for cid in top_k if cid in relevant_ids)
    return {
        f"hit@{k}": 1.0 if found else 0.0,
        f"recall@{k}": found / len(relevant_ids),
        f"precision@{k}": found / k,
    }


def doc_level_metrics(retrieved_files: list[str], relevant_groups: list[set[str]], k: int) -> dict:
    top_k = retrieved_files[:k]
    covered = sum(1 for group in relevant_groups if any(f in group for f in top_k))
    relevant_files = set().union(*relevant_groups)
    return {
        f"hit@{k}": 1.0 if covered else 0.0,
        f"recall@{k}": covered / len(relevant_groups),
        f"precision@{k}": sum(1 for f in top_k if f in relevant_files) / k,
    }


def reciprocal_rank(retrieved: list[str], is_relevant) -> float:
    for idx, item in enumerate(retrieved):
        if is_relevant(item):
            return 1.0 / (idx + 1)
    return 0.0


def bootstrap_ci(values: list[float], n_resamples: int = 2000, confidence: float = 0.95,
                 seed: int = 0) -> tuple[float, float] | None:
    """Khoảng tin cậy percentile bootstrap cho giá trị trung bình — để biết chênh lệch
    giữa 2 lần chạy có vượt nhiễu do cỡ mẫu không."""
    import random

    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    rng = random.Random(seed)
    n = len(clean)
    means = sorted(sum(rng.choices(clean, k=n)) / n for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    return (round(means[int(alpha * n_resamples)], 3), round(means[int((1 - alpha) * n_resamples) - 1], 3))
