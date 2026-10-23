"""
apply_review_sheet.py — Đưa kết quả duyệt trong review_candidates_full.xlsx về lại
evaluation_dataset_full_candidates.json (review.status, review.note, câu hỏi đã sửa,
bỏ nhãn pooled). Chỉ đọc 4 cột người duyệt điền, bỏ qua mọi thay đổi ở cột khác.

Kiểm tra trước khi ghi (sai thì KHÔNG ghi gì): ID phải khớp đúng tập câu trong JSON,
trạng thái phải là approved/rejected/pending.

Chạy: python backend/tests/benchmark/apply_review_sheet.py [--set full|vague]
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openpyxl import load_workbook

from tests.benchmark.export_review_sheet import REVIEW_SHEET, SETS, STATUSES, columns_for


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def read_sheet(path: str, kind: str = "full") -> dict[str, dict]:
    columns = columns_for(kind)
    ws = load_workbook(path, read_only=True, data_only=True)[REVIEW_SHEET]
    keys = [key for _, _, key in columns]
    headers = [_text(c) for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    expected = [header for header, _, _ in columns]
    if headers[:len(expected)] != expected:
        sys.exit("❌ Tiêu đề cột trong sheet đã bị đổi — không xác định được cột người duyệt điền.")

    reviews = {}
    for values in ws.iter_rows(min_row=2, values_only=True):
        row = dict(zip(keys, values))
        qid = _text(row["id"])
        if not qid:
            continue
        if qid in reviews:
            sys.exit(f"❌ ID {qid} xuất hiện 2 lần trong sheet.")
        reviews[qid] = {
            "status": _text(row["status"]).lower() or "pending",
            "question_edit": _text(row["question_edit"]),
            "drop_pooled": _text(row["drop_pooled"]).lower() in ("x", "yes", "có", "1", "true"),
            "note": _text(row["note"]),
            "extra_files": [f.strip() for f in _text(row.get("extra_files")).splitlines() if f.strip()],
        }
    return reviews


def apply(candidates: list[dict], reviews: dict[str, dict]) -> Counter:
    ids = {c["id"] for c in candidates}
    if set(reviews) != ids:
        missing, extra = sorted(ids - set(reviews)), sorted(set(reviews) - ids)
        sys.exit(f"❌ ID không khớp JSON — thiếu: {missing[:10]}, thừa: {extra[:10]}")
    bad = {qid: r["status"] for qid, r in reviews.items() if r["status"] not in STATUSES}
    if bad:
        sys.exit(f"❌ Trạng thái không hợp lệ: {bad}")

    stats = Counter()
    for c in candidates:
        r = reviews[c["id"]]
        c["review"] = {"status": r["status"], "note": r["note"]}
        stats[r["status"]] += 1
        if r["extra_files"] or c.get("extra_relevant_files"):
            c["extra_relevant_files"] = r["extra_files"]
            stats["extra_files"] += len(r["extra_files"])
        if r["question_edit"] and r["question_edit"] != c["question"]:
            c.setdefault("question_original", c["question"])
            c["question"] = r["question_edit"]
            stats["question_edited"] += 1
        if r["drop_pooled"]:
            before = len(c["relevant_chunks"])
            c["relevant_chunks"] = [x for x in c["relevant_chunks"] if x["via"] != "pooled"]
            if len(c["relevant_chunks"]) < before:
                kept = {x["filename"] for x in c["relevant_chunks"]}
                c["expected_sources"] = [s for s in c["expected_sources"] if s["filename"] in kept]
                c["flags"] = [f for f in c["flags"] if f != "answer_also_in_other_document"]
                stats["pooled_dropped"] += 1
    return stats


def run(kind: str = "full"):
    candidates_path = SETS[kind]["candidates"]
    with open(candidates_path, encoding="utf-8") as f:
        data = json.load(f)
    stats = apply(data["candidates"], read_sheet(SETS[kind]["sheet"], kind))
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã cập nhật {candidates_path}")
    print(f"   {dict(stats)}")
    if stats["pending"]:
        print(f"⚠️  Còn {stats['pending']} câu pending — finalize sẽ BỎ QUA các câu này.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", dest="kind", choices=sorted(SETS), default="full", help="Bộ câu hỏi")
    run(kind=parser.parse_args().kind)
