"""
test_benchmark_metrics.py — Unit test cho metric của benchmark tổng quát
(tests/benchmark/metrics.py: chunk_level_metrics, doc_level_metrics, reciprocal_rank,
bootstrap_ci) và chia dev/test (finalize_benchmark_dataset.stratified_split).
KHÔNG cần Ollama/Gemini/ChromaDB.

Chạy: python backend/tests/test_benchmark_metrics.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.benchmark.metrics import (
    bootstrap_ci,
    chunk_level_metrics,
    doc_level_metrics,
    reciprocal_rank,
)
from tests.benchmark.finalize_benchmark_dataset import stratified_split


def test_chunk_level_metrics():
    retrieved = ["c9", "c1", "c8", "c2", "c7"]
    relevant = {"c1", "c2"}
    m3 = chunk_level_metrics(retrieved, relevant, 3)
    assert m3 == {"hit@3": 1.0, "recall@3": 0.5, "precision@3": 1 / 3}
    m5 = chunk_level_metrics(retrieved, relevant, 5)
    assert m5["recall@5"] == 1.0 and m5["precision@5"] == 2 / 5
    assert chunk_level_metrics(retrieved, relevant, 1)["hit@1"] == 0.0
    # Ít kết quả hơn k: precision vẫn chia cho k (không thưởng retriever trả thiếu)
    assert chunk_level_metrics(["c1"], relevant, 3)["precision@3"] == 1 / 3
    print("✅ chunk_level_metrics: hit/recall/precision đúng, chia cho k cố định")


def test_doc_level_metrics_groups_duplicates():
    # 45.signed.pdf và Boluatlaodong2019.pdf là 1 nhóm trùng: lấy bản nào cũng đủ
    groups = [{"45.signed.pdf", "Boluatlaodong2019.pdf"}, {"luat-so-125.pdf"}]
    retrieved = ["Boluatlaodong2019.pdf", "Boluatlaodong2019.pdf", "other.pdf", "luat-so-125.pdf"]
    m3 = doc_level_metrics(retrieved, groups, 3)
    assert m3["hit@3"] == 1.0
    assert m3["recall@3"] == 0.5          # mới phủ 1/2 nhóm
    assert m3["precision@3"] == 2 / 3     # 2/3 chunk thuộc tài liệu đúng
    assert doc_level_metrics(retrieved, groups, 4)["recall@4"] == 1.0
    # Lấy CẢ 2 bản trùng không được tính là phủ 2 nhóm
    dup_only = doc_level_metrics(["45.signed.pdf", "Boluatlaodong2019.pdf"], groups, 2)
    assert dup_only["recall@2"] == 0.5
    print("✅ doc_level_metrics: gom nhóm tài liệu trùng, recall theo nhóm")


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], lambda x: x == "c") == 1 / 3
    assert reciprocal_rank(["a", "b"], lambda x: x == "z") == 0.0
    assert reciprocal_rank([], lambda x: True) == 0.0
    print("✅ reciprocal_rank đúng, kể cả danh sách rỗng")


def test_bootstrap_ci():
    values = [1.0] * 80 + [0.0] * 20
    lo, hi = bootstrap_ci(values, seed=1)
    assert lo < 0.8 < hi
    assert 0.65 < lo and hi < 0.95       # n=100, p=0.8 → ±~0.08
    assert bootstrap_ci([1.0]) is None
    assert bootstrap_ci([1.0, 1.0, None]) == (1.0, 1.0)
    assert bootstrap_ci(values, seed=1) == (lo, hi)  # tái lập được với cùng seed
    print(f"✅ bootstrap_ci: n=100 p=0.8 → [{lo}, {hi}], tái lập được")


def test_stratified_split():
    items = [{"id": f"q{i:03d}", "stratum": "legal" if i < 20 else "xlsx"} for i in range(30)]
    split = stratified_split(items)
    test_legal = sum(split[f"q{i:03d}"] == "test" for i in range(20))
    test_xlsx = sum(split[f"q{i:03d}"] == "test" for i in range(20, 30))
    assert (test_legal, test_xlsx) == (6, 3)  # 30% mỗi nhóm
    assert stratified_split(items) == split   # seed cố định
    print("✅ stratified_split: 30% test trong TỪNG nhóm, tái lập được")


def test_gemini_rate_limiter_respects_rpm_and_tpm():
    """Đồng hồ giả: không request nào vượt 15 req hoặc 250k token trong BẤT KỲ cửa sổ 60s."""
    import tests.benchmark.ragas_metrics as rm

    clock = {"now": 1000.0}
    real_monotonic, real_sleep = rm.time.monotonic, rm.time.sleep
    rm.time.monotonic = lambda: clock["now"]
    rm.time.sleep = lambda s: clock.__setitem__("now", clock["now"] + s)
    rm._call_log.clear()
    try:
        starts = []
        for _ in range(40):                      # nhẹ: chỉ RPM giới hạn
            rm._rate_limit_wait(1_000)
            starts.append((clock["now"], 1_000))
            clock["now"] += 0.5                  # round-trip giả
        for _ in range(10):                      # nặng: 100k token/request → TPM giới hạn
            rm._rate_limit_wait(100_000)
            starts.append((clock["now"], 100_000))
            clock["now"] += 0.5
        for t, _ in starts:
            window = [tok for s, tok in starts if t <= s < t + 60]
            assert len(window) <= 15, f"{len(window)} request trong 60s"
            assert sum(window) <= 250_000, f"{sum(window)} token trong 60s"
        assert starts[14][0] - starts[0][0] < 10  # 15 request đầu đi liền, không giãn cách thừa
        assert starts[15][0] - starts[0][0] >= 60  # request thứ 16 chờ hết cửa sổ
    finally:
        rm.time.monotonic, rm.time.sleep = real_monotonic, real_sleep
        rm._call_log.clear()
    print("✅ Gemini rate limiter: ≤15 req và ≤250k token trong mọi cửa sổ 60s")


def test_pooling_parse_indices_accepts_strings():
    from tests.benchmark.generate_benchmark_questions import parse_indices

    assert parse_indices(["9"], 18) == {9}            # Gemini trả chuỗi — case thật đã gặp
    assert parse_indices([1, "3", " 4 ", "[5]"], 18) == {1, 3, 4, 5}
    assert parse_indices([0, 19, "x", None, 2.0], 18) == {2}  # ngoài khoảng / rác bị bỏ
    assert parse_indices([], 18) == set()
    print("✅ parse_indices nhận cả số lẫn chuỗi, loại chỉ số ngoài khoảng")


if __name__ == "__main__":
    test_pooling_parse_indices_accepts_strings()
    test_gemini_rate_limiter_respects_rpm_and_tpm()
    test_chunk_level_metrics()
    test_doc_level_metrics_groups_duplicates()
    test_reciprocal_rank()
    test_bootstrap_ci()
    test_stratified_split()
    print("\n🎉 TẤT CẢ TEST METRICS PASS")
