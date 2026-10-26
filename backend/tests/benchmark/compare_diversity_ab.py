"""
compare_diversity_ab.py — Cổng M4 (multi_doc_selection_plan.md §3): A/B killswitch RERANK_DIVERSITY_ENABLED.

Đọc report chạy `run_benchmark_full.py --diversity`: mỗi câu có cả `reranked` (BẬT) lẫn `reranked_off`
(thứ tự điểm thuần = TẮT) tính trên CÙNG ứng viên / câu con SLM → so cặp không lẫn nhiễu sinh câu con.

Tiêu chí (mỗi report × chế độ):
  - nhóm multi_doc all_required@3: ≥ 0.6 ở slm, ≥ 0.45 ở none/fast (chỉ khi bộ có nhóm này);
  - 0 câu đúng→sai doc hit@3 (≤ 1 nếu kiểm tay nhãn hẹp — script chỉ liệt kê, không tự miễn);
  - không stratum / category nào giảm doc hit@3;
  - câu kích hoạt nhưng chỉ cần 1 tài liệu: chunk hit@3 giảm ≤ 1 câu;
  - câu không kích hoạt: top-3 trùng hệt bản tắt (trong cùng lượt; thêm --off-report để đối chiếu lượt TẮT riêng,
    chỉ có nghĩa ở none/fast vì slm sinh câu con khác mỗi lượt);
  - thời gian chọn đa dạng ≤ 50 ms.

Chạy:
  python backend/tests/benchmark/compare_diversity_ab.py <report_on.json> [--off-report <report_off.json>]
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.benchmark.metrics import bootstrap_ci

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
MULTI_DOC_CATEGORY = "multi_doc"
MIN_ALL_REQUIRED = {"slm": 0.6, "none": 0.45, "fast": 0.45}
MAX_SELECT_MS = 50
MAX_SINGLE_DOC_CHUNK_LOSS = 1


def _load(path: str) -> dict:
    path = path if os.path.isabs(path) or os.path.exists(path) else os.path.join(REPORTS_DIR, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _metric(record: dict, stage: str, level: str, key: str):
    return record["metrics"][stage][level].get(key)


def _mean(values) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def sign_test(better: int, worse: int) -> float:
    """p hai phía của sign test (bỏ câu hòa) — binomial chính xác p=0.5."""
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(better, worse) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _pair(details: list[dict], level: str, key: str) -> tuple[float | None, float | None, list[dict], list[dict]]:
    """(TB tắt, TB bật, câu tốt lên, câu tệ đi) cho 1 chỉ số; bỏ câu có giá trị None."""
    pairs = [(d, _metric(d, "reranked_off", level, key), _metric(d, "reranked", level, key)) for d in details]
    pairs = [(d, off, on) for d, off, on in pairs if off is not None and on is not None]
    better = [d for d, off, on in pairs if on > off]
    worse = [d for d, off, on in pairs if on < off]
    return _mean(off for _, off, _ in pairs), _mean(on for _, _, on in pairs), better, worse


def compare(name: str, report: dict, mode: str, off_report: dict | None) -> tuple[bool, list[str]]:
    lines, ok = [], True

    def check(cond: bool, msg: str):
        nonlocal ok
        ok &= cond
        lines.append(f"  {'✅' if cond else '❌'} {msg}")

    details = report["modes"][mode]["details"]
    lines.append(f"\n=== {name} · chế độ {mode} · {len(details)} câu ===")
    if not report["run_config"].get("diversity_enabled") or "reranked_off" not in details[0]["metrics"]:
        return False, lines + ["  ❌ report không chạy --diversity (thiếu reranked_off)"]

    triggered = [d for d in details if d["query_plan"].get("diverse_selection")]
    lines.append(f"  ·  câu kích hoạt chọn đa dạng: {len(triggered)}/{len(details)}")

    # Nhóm cần nhiều tài liệu
    multi = [d for d in details if d["category"] == MULTI_DOC_CATEGORY]
    if multi:
        lines.append(f"  ·  nhóm {MULTI_DOC_CATEGORY} ({len(multi)} câu, kích hoạt "
                     f"{sum(d in triggered for d in multi)}):")
        for key in ("all_required@3", "all_required@5", "hit@3", "mrr"):
            off, on, better, worse = _pair(multi, "doc", key)
            diffs = [_metric(d, "reranked", "doc", key) - _metric(d, "reranked_off", "doc", key) for d in multi
                     if _metric(d, "reranked_off", "doc", key) is not None]
            ci = bootstrap_ci(diffs)
            lines.append(f"       doc {key:<15} {off:.3f} → {on:.3f}  chênh CI95 {ci}  "
                         f"+{len(better)}/−{len(worse)}  sign p={sign_test(len(better), len(worse)):.3f}")
        _, on, _, worse = _pair(multi, "doc", "all_required@3")
        threshold = MIN_ALL_REQUIRED.get(mode, 0.45)
        check(on >= threshold, f"{MULTI_DOC_CATEGORY} all_required@3 = {on:.3f} (≥ {threshold})")
        if worse:
            lines.append("       ↓ all_required@3: " + ", ".join(d["id"] for d in worse))

    # Toàn bộ: doc hit@3 / chunk hit@3
    for level in ("doc", "chunk"):
        off, on, better, worse = _pair(details, level, "hit@3")
        lines.append(f"  ·  {level} hit@3 toàn bộ {off:.3f} → {on:.3f}  (+{len(better)}/−{len(worse)})")
        if level == "doc":
            check(not worse, f"0 câu đúng→sai doc hit@3 (đúng→sai {len(worse)}, sai→đúng {len(better)})")
            for d in worse:
                lines.append(f"       ↓ {d['id']} [{d['category']}] {d['question'][:90]}")

    for field in ("stratum", "category"):
        groups = defaultdict(list)
        for d in details:
            groups[d[field]].append(d)
        dropped = []
        for group_name, group in sorted(groups.items()):
            off, on, _, _ = _pair(group, "doc", "hit@3")
            if off is not None and on < off:
                dropped.append(f"{group_name} {off:.3f}→{on:.3f}")
        check(not dropped, f"không {field} nào giảm doc hit@3" + (f": {', '.join(dropped)}" if dropped else ""))

    # Câu kích hoạt nhưng chỉ cần 1 tài liệu: chọn đa dạng có thể đẩy chunk đáp án thứ 2 cùng tài liệu ra
    # (không có required_doc_groups → all_required = None)
    single = [d for d in triggered if _metric(d, "reranked", "doc", "all_required@3") is None]
    _, _, better, worse = _pair(single, "chunk", "hit@3")
    check(len(worse) <= MAX_SINGLE_DOC_CHUNK_LOSS,
          f"câu kích hoạt 1-tài-liệu ({len(single)}): chunk hit@3 −{len(worse)}/+{len(better)} "
          f"(giảm ≤ {MAX_SINGLE_DOC_CHUNK_LOSS})" + (f": {', '.join(d['id'] for d in worse)}" if worse else ""))

    # Câu không kích hoạt: top-3 trùng hệt bản tắt
    not_triggered = [d for d in details if not d["query_plan"].get("diverse_selection")]
    same_run_diff = [d["id"] for d in not_triggered if d["reranked_top"] != d["reranked_top_off"]]
    check(not same_run_diff, f"câu không kích hoạt ({len(not_triggered)}): top-3 trùng bản tắt cùng lượt"
          + (f" — lệch {same_run_diff}" if same_run_diff else ""))
    if off_report is not None and mode in off_report["modes"]:
        off_by_id = {d["id"]: d for d in off_report["modes"][mode]["details"]}
        cross = [d["id"] for d in not_triggered
                 if d["id"] in off_by_id and d["reranked_top"] != off_by_id[d["id"]]["reranked_top"]]
        missing = sum(d["id"] not in off_by_id for d in not_triggered)
        note = " (slm: câu con sinh lại mỗi lượt, lệch là nhiễu SLM)" if mode == "slm" else ""
        cond = not cross and not missing
        if mode == "slm":
            lines.append(f"  ·  so lượt TẮT riêng: {len(cross)} câu không kích hoạt lệch top-3{note}")
        else:
            check(cond, f"câu không kích hoạt: top-3 trùng lượt TẮT riêng ({len(not_triggered) - missing} câu đối chiếu"
                  + (f", lệch {cross}" if cross else "") + (f", thiếu {missing}" if missing else "") + ")")

    select_ms = [d["query_plan"].get("select_ms") or 0.0 for d in triggered]
    if select_ms:
        check(max(select_ms) <= MAX_SELECT_MS,
              f"thời gian chọn đa dạng TB {_mean(select_ms):.3f} ms, max {max(select_ms):.3f} ms (≤ {MAX_SELECT_MS} ms)")
    return ok, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="+", help="Report chạy --diversity (có reranked_off)")
    parser.add_argument("--off-report", action="append", default=[],
                        help="Report lượt TẮT riêng, cùng thứ tự với report (tùy chọn, đối chiếu câu không kích hoạt)")
    parser.add_argument("--mode", action="append", help="Chế độ cần so (mặc định: mọi chế độ trong report)")
    args = parser.parse_args()
    all_ok = True
    for i, path in enumerate(args.report):
        report = _load(path)
        off_report = _load(args.off_report[i]) if i < len(args.off_report) and args.off_report[i] else None
        for mode in args.mode or list(report["modes"]):
            if mode not in report["modes"]:
                continue
            ok, lines = compare(os.path.basename(path), report, mode, off_report)
            all_ok &= ok
            print("\n".join(lines))
    print(f"\n{'✅ ĐẠT' if all_ok else '❌ CHƯA ĐẠT'} cổng M4 (multi_doc_selection_plan.md §3.3)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
