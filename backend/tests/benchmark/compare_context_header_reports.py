"""
compare_context_header_reports.py — Cổng quyết định S5 (contextual_header_rollout_plan.md §4.3):
so report benchmark KHÔNG header (baseline) với report CÓ header, theo từng split/chế độ.

Tiêu chí chấp nhận (mỗi split):
  - rerank doc hit@3 toàn bộ ≥ 0.95;
  - KHÔNG nhóm (stratum) nào giảm doc hit@3 sau rerank; financial_xlsx = 1.0;
  - financial_report ≥ 0.9;
  - latency rerank trung bình tăng ≤ 1,5 s/câu.
Kèm danh sách câu đổi kết quả (đúng→sai / sai→đúng) ở mức tài liệu để soi tác dụng phụ.

Chạy:
  python backend/tests/benchmark/compare_context_header_reports.py \\
      --pair dev:<baseline.json>:<header.json> --pair test:<baseline.json>:<header.json> [--mode none]
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
MIN_OVERALL = 0.95
MIN_FINANCIAL_REPORT = 0.9
MAX_RERANK_LATENCY_INCREASE_MS = 1500


def _load(path: str) -> dict:
    path = path if os.path.isabs(path) or os.path.exists(path) else os.path.join(REPORTS_DIR, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _doc_hit3(record: dict) -> float | None:
    """doc hit@3 sau rerank của 1 câu (details[i].metrics.reranked.doc)."""
    return (((record.get("metrics") or {}).get("reranked") or {}).get("doc") or {}).get("hit@3")


def compare(split: str, base: dict, ctx: dict, mode: str) -> tuple[bool, list[str]]:
    lines, ok = [], True
    if mode not in base["modes"] or mode not in ctx["modes"]:
        return False, [f"❌ [{split}/{mode}] thiếu chế độ trong 1 trong 2 report"]
    b, c = base["modes"][mode]["summary"], ctx["modes"][mode]["summary"]

    def check(cond: bool, msg: str):
        nonlocal ok
        ok &= cond
        lines.append(f"  {'✅' if cond else '❌'} {msg}")

    lines.append(f"\n=== {split} · chế độ {mode} · {c['n']} câu ===")
    if not ctx["run_config"].get("context_header_enabled"):
        check(False, "report có header KHÔNG bật --context-header")
    bo, co = b["reranked"]["doc"]["hit@3"]["mean"], c["reranked"]["doc"]["hit@3"]["mean"]
    check(co >= MIN_OVERALL, f"rerank doc hit@3 toàn bộ {bo:.3f} → {co:.3f} (≥ {MIN_OVERALL})")
    for stratum in sorted(set(b["by_stratum"]) | set(c["by_stratum"])):
        bs = b["by_stratum"].get(stratum, {}).get("doc_hit@3_reranked")
        cs = c["by_stratum"].get(stratum, {}).get("doc_hit@3_reranked")
        if bs is None or cs is None:
            check(False, f"nhóm {stratum}: thiếu số liệu ({bs} → {cs})")
            continue
        target = " (= 1.0)" if stratum == "financial_xlsx" else (f" (≥ {MIN_FINANCIAL_REPORT})" if stratum == "financial_report" else "")
        cond = cs >= bs
        if stratum == "financial_xlsx":
            cond &= cs == 1.0
        if stratum == "financial_report":
            cond &= cs >= MIN_FINANCIAL_REPORT
        check(cond, f"nhóm {stratum:<17} {bs:.3f} → {cs:.3f}{target}")
    bl, cl = b["latency"]["rerank_ms"]["mean"], c["latency"]["rerank_ms"]["mean"]
    check(cl - bl <= MAX_RERANK_LATENCY_INCREASE_MS,
          f"latency rerank TB {bl / 1000:.2f}s → {cl / 1000:.2f}s (tăng ≤ {MAX_RERANK_LATENCY_INCREASE_MS / 1000:.1f}s)")
    lines.append(f"  ·  chunk hit@3 sau rerank {b['reranked']['chunk']['hit@3']['mean']:.3f} → "
                 f"{c['reranked']['chunk']['hit@3']['mean']:.3f} · doc MRR retrieval "
                 f"{b['retrieval']['doc']['mrr']['mean']:.3f} → {c['retrieval']['doc']['mrr']['mean']:.3f}")

    base_by_id = {d["id"]: d for d in base["modes"][mode]["details"]}
    worse, better = [], []
    for d in ctx["modes"][mode]["details"]:
        before, after = _doc_hit3(base_by_id.get(d["id"], {})), _doc_hit3(d)
        if before is None or after is None:
            continue
        if before > after:
            worse.append(d)
        elif after > before:
            better.append(d)
    lines.append(f"  ·  câu đổi kết quả mức tài liệu: sai→đúng {len(better)}, đúng→sai {len(worse)}")
    for d in worse:
        lines.append(f"     ↓ {d['id']} [{d.get('stratum')}] {d.get('question', '')[:90]}")
    return ok, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True, help="split:baseline.json:header.json")
    parser.add_argument("--mode", action="append", help="Chế độ cần so (mặc định: none và fast nếu có)")
    args = parser.parse_args()
    all_ok = True
    for pair in args.pair:
        split, base_path, ctx_path = pair.split(":", 2) if pair.count(":") == 2 else (None, None, None)
        if not split:
            parser.error(f"--pair sai định dạng: {pair}")
        base, ctx = _load(base_path), _load(ctx_path)
        modes = args.mode or [m for m in ("none", "fast") if m in ctx["modes"]]
        for mode in modes:
            ok, lines = compare(split, base, ctx, mode)
            all_ok &= ok
            print("\n".join(lines))
    print(f"\n{'✅ ĐẠT' if all_ok else '❌ CHƯA ĐẠT'} tiêu chí §4.3 — "
          f"{'header không làm nhóm nào tệ đi' if all_ok else 'DỪNG, không triển khai tiếp'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
