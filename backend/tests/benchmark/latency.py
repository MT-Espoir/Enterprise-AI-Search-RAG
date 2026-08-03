"""
latency.py — Tiện ích đo hiệu năng (response time) dùng CHUNG cho các script benchmark
(run_full_benchmark.py, run_full_benchmark_legal.py).

Vì sao tách riêng module này thay vì đo inline trong từng script:
  - Tránh lặp lại logic tính percentile/tổng hợp ở 2 (hoặc nhiều hơn) benchmark gần
    giống nhau — sửa 1 chỗ, cả 2 cùng đúng.
  - Có thể unit-test độc lập (test_benchmark_latency.py) mà không cần chạy cả pipeline.

Nguyên tắc đo cho ra số CHÂN THẬT (không đẹp giả tạo):
  1. Chỉ đo các stage THUỘC HỆ THỐNG (preprocessing/retrieval/rerank/generation).
     KHÔNG bao giờ tính thời gian gọi Gemini judge — judge là công cụ chấm điểm eval,
     không phải thành phần production. Người dùng thật không phải chờ judge.
  2. Warm-up TRƯỚC khi đo (xem warm_up_pipeline) — nạp model vào RAM/VRAM (BGE-M3,
     bge-reranker-v2-m3, model Ollama) để câu đầu tiên không gánh cold-start làm
     lệch số liệu.
  3. Báo cáo PERCENTILE (p50/p95/p99), không chỉ trung bình — trung bình che giấu
     tail latency, thứ quan trọng nhất khi đánh giá trải nghiệm thật của hệ thống.

Đơn vị: mọi số latency đều tính bằng mili-giây (ms), làm tròn 2 chữ số.
"""
import math
import time
from contextlib import contextmanager

# Các stage khớp ĐÚNG với breakdown của observability production
# (app/core/observability/request_tracer.py) — dùng cùng tên để báo cáo benchmark
# đọc được liền mạch với log production, không phải "một hệ đo khác".
STAGE_KEYS = ("preprocessing_ms", "retrieval_ms", "rerank_ms", "generation_ms")
LATENCY_KEYS = STAGE_KEYS + ("end_to_end_ms",)


class StageTimer:
    """
    Đo thời gian từng stage bằng time.perf_counter() (đồng hồ đơn điệu, không bị
    ảnh hưởng khi giờ hệ thống nhảy — chuẩn để đo khoảng thời gian).

    Dùng:
        timer = StageTimer()
        with timer.stage("retrieval_ms"):
            ...
        with timer.stage("generation_ms"):
            ...
        timer.as_record()  -> {"retrieval_ms": .., "generation_ms": .., "end_to_end_ms": ..}
    """

    def __init__(self):
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Cộng dồn nếu 1 stage được đo nhiều lần trong cùng câu hỏi (vd multi-query
            # retrieve gọi retriever nhiều lần) — phản ánh đúng TỔNG thời gian stage đó.
            self.stages[name] = round(self.stages.get(name, 0.0) + elapsed_ms, 2)

    def as_record(self) -> dict:
        """Trả về dict latency cho 1 câu hỏi: đủ 4 stage (null nếu stage không chạy)
        + end_to_end_ms = tổng các stage đã đo."""
        record = {key: self.stages.get(key) for key in STAGE_KEYS}
        measured = [v for v in self.stages.values() if v is not None]
        record["end_to_end_ms"] = round(sum(measured), 2) if measured else None
        return record


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Percentile theo nội suy tuyến tính (linear interpolation) — cùng phương pháp
    numpy.percentile mặc định, nhưng không cần dependency numpy. p tính theo %."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _stats(values: list[float]) -> dict | None:
    """Thống kê 1 stage: mean/p50/p95/p99/min/max + n (số mẫu hợp lệ)."""
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    return {
        "n": len(clean),
        "mean": round(sum(clean) / len(clean), 2),
        "p50": round(_percentile(clean, 50), 2),
        "p95": round(_percentile(clean, 95), 2),
        "p99": round(_percentile(clean, 99), 2),
        "min": round(clean[0], 2),
        "max": round(clean[-1], 2),
    }


def summarize_latency(latency_records: list[dict]) -> dict:
    """
    Tổng hợp latency toàn bộ benchmark từ list per-question latency dict (mỗi phần tử
    là output của StageTimer.as_record()). Trả về {stage: {mean,p50,p95,p99,min,max,n}}.
    """
    result = {}
    for key in LATENCY_KEYS:
        result[key] = _stats([rec.get(key) for rec in latency_records if rec])
    return result


def warm_up_pipeline(query_processor, retriever, reranker, generator, warm_question: str) -> float:
    """
    Chạy 1 lượt "nháp" qua đủ 4 stage để nạp toàn bộ model vào bộ nhớ TRƯỚC khi đo thật
    — nếu không, câu đầu tiên sẽ gánh cold-start (load BGE-M3 + bge-reranker-v2-m3 +
    model Ollama), làm p95/p99 và cả trung bình bị thổi phồng phi thực tế.

    KHÔNG ghi lại kết quả câu này. Trả về thời gian warm-up (giây) để in ra tham khảo.
    Fail-open: lỗi ở bước warm-up (vd corpus rỗng) KHÔNG làm dừng benchmark.
    """
    t0 = time.perf_counter()
    try:
        plan = query_processor.process(warm_question)
        vector_query = plan.rewritten_query if plan.rewrite else None
        candidates = retriever.retrieve(warm_question, vector_query=vector_query)
        if candidates:
            top_chunks = reranker.rerank(warm_question, candidates)
            generator.generate(warm_question, top_chunks)
    except Exception as exc:  # noqa: BLE001 — warm-up không được phép làm hỏng benchmark
        print(f"   ⚠️  Warm-up gặp lỗi (bỏ qua, không ảnh hưởng số đo): {exc}")
    return round(time.perf_counter() - t0, 2)
