"""
test_benchmark_latency.py — Unit test cho tiện ích đo hiệu năng benchmark
(tests/benchmark/latency.py). KHÔNG cần Ollama/Gemini/ChromaDB — chỉ test logic
đo thời gian + percentile + tổng hợp thuần.

Chạy: python backend/tests/test_benchmark_latency.py
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.benchmark.latency import (
    StageTimer,
    summarize_latency,
    _percentile,
    _stats,
)


def test_stage_timer_measures_and_totals():
    timer = StageTimer()
    with timer.stage("retrieval_ms"):
        time.sleep(0.02)
    with timer.stage("generation_ms"):
        time.sleep(0.03)
    rec = timer.as_record()
    # Đủ 4 key stage (2 stage không đo = None)
    assert rec["preprocessing_ms"] is None
    assert rec["rerank_ms"] is None
    # 2 stage đã đo phải > 0 và hợp lý (sleep 20ms/30ms, cho biên độ rộng vì CI chậm)
    assert rec["retrieval_ms"] >= 15
    assert rec["generation_ms"] >= 25
    # end_to_end = tổng các stage đã đo
    assert abs(rec["end_to_end_ms"] - (rec["retrieval_ms"] + rec["generation_ms"])) < 0.01
    print("✅ StageTimer đo + tổng hợp end_to_end đúng")


def test_stage_timer_accumulates_repeated_stage():
    """Stage đo nhiều lần (vd multi-query gọi retrieve nhiều lần) phải CỘNG DỒN."""
    timer = StageTimer()
    with timer.stage("retrieval_ms"):
        time.sleep(0.01)
    with timer.stage("retrieval_ms"):
        time.sleep(0.01)
    rec = timer.as_record()
    assert rec["retrieval_ms"] >= 15, rec["retrieval_ms"]  # ~20ms tổng, không phải ~10ms
    print("✅ StageTimer cộng dồn stage lặp lại đúng")


def test_percentile_interpolation():
    vals = [10, 20, 30, 40, 50]
    assert _percentile(vals, 50) == 30
    assert _percentile(vals, 0) == 10
    assert _percentile(vals, 100) == 50
    # p95 nội suy tuyến tính giữa 40 và 50: (5-1)*0.95=3.8 -> 40 + (50-40)*0.8 = 48
    assert abs(_percentile(vals, 95) - 48) < 1e-9
    # 1 phần tử -> chính nó
    assert _percentile([7], 95) == 7
    # rỗng -> None
    assert _percentile([], 95) is None
    print("✅ _percentile nội suy đúng (khớp numpy.percentile mặc định)")


def test_stats_ignores_none():
    s = _stats([10, None, 30, None, 50])
    assert s["n"] == 3
    assert s["mean"] == 30
    assert s["min"] == 10
    assert s["max"] == 50
    assert _stats([None, None]) is None
    assert _stats([]) is None
    print("✅ _stats bỏ qua None, min/max/mean/n đúng")


def test_summarize_latency_across_records():
    records = [
        {"preprocessing_ms": 5, "retrieval_ms": 100, "rerank_ms": 50, "generation_ms": 800, "end_to_end_ms": 955},
        {"preprocessing_ms": None, "retrieval_ms": 120, "rerank_ms": 60, "generation_ms": 900, "end_to_end_ms": 1080},
        {"preprocessing_ms": None, "retrieval_ms": None, "rerank_ms": None, "generation_ms": 700, "end_to_end_ms": 700},
    ]
    summary = summarize_latency(records)
    # preprocessing chỉ có 1 mẫu hợp lệ
    assert summary["preprocessing_ms"]["n"] == 1
    # generation có đủ 3 mẫu
    assert summary["generation_ms"]["n"] == 3
    assert summary["generation_ms"]["min"] == 700
    assert summary["generation_ms"]["max"] == 900
    # end_to_end đủ 3 mẫu, max = 1080
    assert summary["end_to_end_ms"]["max"] == 1080
    print("✅ summarize_latency tổng hợp per-stage đúng (kể cả stage thiếu mẫu)")


if __name__ == "__main__":
    test_stage_timer_measures_and_totals()
    test_stage_timer_accumulates_repeated_stage()
    test_percentile_interpolation()
    test_stats_ignores_none()
    test_summarize_latency_across_records()
    print("\n🎉 TẤT CẢ TEST LATENCY PASS")
