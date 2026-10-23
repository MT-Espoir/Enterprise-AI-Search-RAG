"""
finalize_benchmark_dataset.py — Chốt evaluation_dataset_full.json từ các ứng viên ĐÃ
DUYỆT trong evaluation_dataset_full_candidates.json.

Cách duyệt (sửa trực tiếp file candidates):
  - review.status = "approved"  → đưa vào benchmark (có thể sửa "question" trước khi duyệt)
  - review.status = "rejected"  → loại
  - Có thể xóa/thêm phần tử trong relevant_chunks nếu pooling gán sai.
Câu còn "pending" bị BỎ QUA — không bao giờ tự động coi là đúng.

Xử lý khi chốt:
  1. Bổ sung chunk TRÙNG NGUYÊN VĂN ở tài liệu khác (vd 45.signed.pdf ↔
     Boluatlaodong2019.pdf trùng 309/309 chunk) vào relevant_chunk_ids — retriever lấy
     bản sao nào cũng được tính đúng.
  2. relevant_doc_groups: gom filename của các chunk đúng theo nhóm trùng nội dung
     (equivalence_groups trong benchmark_sample_full.json + MANUAL_EQUIVALENT_DOCS).
  3. Chia dev/test PHÂN TẦNG theo stratum (seed cố định): dev để tinh chỉnh hệ thống,
     test chỉ dùng cho lần đánh giá cuối — tránh overfit vào chính bộ đo.

Chạy: python backend/tests/benchmark/finalize_benchmark_dataset.py
"""
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

import chromadb

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BENCH_DIR, "benchmark_chromadb_full")
COLLECTION_NAME = "benchmark_full"
MANIFEST_PATH = os.path.join(BENCH_DIR, "snapshot_manifest_full.json")
SAMPLE_PATH = os.path.join(BENCH_DIR, "benchmark_sample_full.json")
CANDIDATES_PATH = os.path.join(BENCH_DIR, "evaluation_dataset_full_candidates.json")
OUTPUT_PATH = os.path.join(BENCH_DIR, "evaluation_dataset_full.json")

SEED = 20261022
TEST_RATIO = 0.3

# Cùng 1 văn bản nhưng khác bản trích xuất (bản ký số PDF / bản DOCX / bản in web) nên trùng
# chunk NGUYÊN VĂN rất ít — sample_benchmark_docs.py không tự phát hiện được. Xác minh THỦ
# CÔNG bằng phần đầu văn bản (2026-10-22). KHÔNG gom theo ngưỡng trùng tự động: luat-so-125.pdf
# (bản HỢP NHẤT Bộ luật Lao động có sửa đổi) trùng 68% với 45.signed.pdf nhưng là phiên bản
# khác — gom nhầm sẽ che mất đúng vấn đề xung đột phiên bản văn bản.
MANUAL_EQUIVALENT_DOCS = [
    ["59.signed.pdf", "luatdoanhnghiep.pdf"],                       # Luật Doanh nghiệp 59/2020/QH14
    ["145.signed.pdf", "1452020nd-cp_25102023165145.docx"],         # Nghị định 145/2020/NĐ-CP
]


def _norm_hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def stratified_split(items: list[dict]) -> dict[str, str]:
    rng = random.Random(SEED)
    by_stratum = defaultdict(list)
    for item in items:
        by_stratum[item["stratum"]].append(item["id"])
    split = {}
    for stratum in sorted(by_stratum):
        ids = sorted(by_stratum[stratum])
        rng.shuffle(ids)
        n_test = round(len(ids) * TEST_RATIO)
        for i, qid in enumerate(ids):
            split[qid] = "test" if i < n_test else "dev"
    return split


def run():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        sample = json.load(f)
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        candidates = json.load(f)
    if candidates["snapshot_fingerprint"] != manifest["fingerprint"]:
        sys.exit("❌ Candidates được sinh từ snapshot khác snapshot hiện tại.")

    status_count = Counter(c["review"]["status"] for c in candidates["candidates"])
    approved = [c for c in candidates["candidates"] if c["review"]["status"] == "approved"]
    print(f"Trạng thái duyệt: {dict(status_count)}")
    if not approved:
        sys.exit("❌ Chưa có câu nào review.status = \"approved\".")

    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    hash_of = {}
    twins = defaultdict(list)
    filename_of = {}
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        h = _norm_hash(text)
        hash_of[cid] = h
        twins[h].append(cid)
        filename_of[cid] = meta["filename"]

    group_of = {}
    for group in sample["equivalence_groups"] + MANUAL_EQUIVALENT_DOCS:
        for f in group:
            group_of[f] = tuple(sorted(group))

    items = []
    added_twins = 0
    for c in approved:
        relevant = {r["chunk_id"] for r in c["relevant_chunks"]}
        expanded = set(relevant)
        for cid in relevant:
            expanded.update(twins[hash_of[cid]])
        added_twins += len(expanded) - len(relevant)
        groups = {group_of.get(filename_of[cid], (filename_of[cid],)) for cid in expanded}
        items.append({
            "id": c["id"],
            "question": c["question"],
            "reference_answer": c["reference_answer"],
            "category": c["category"],
            "stratum": c["stratum"],
            "relevant_chunk_ids": sorted(expanded),
            "relevant_doc_groups": sorted(sorted(g) for g in groups),
            "source_chunk_ids": [s["chunk_id"] for s in c["source_chunks"]],
        })

    split = stratified_split(items)
    for item in items:
        item["split"] = split[item["id"]]

    output = {
        "snapshot_fingerprint": manifest["fingerprint"],
        "generator": candidates["generator"],
        "split_seed": SEED,
        "test_ratio": TEST_RATIO,
        "counts": {
            "total": len(items),
            "by_split": dict(Counter(i["split"] for i in items)),
            "by_stratum": dict(Counter(i["stratum"] for i in items)),
            "by_category": dict(Counter(i["category"] for i in items)),
        },
        "questions": items,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(items)} câu hỏi — {output['counts']}")
    print(f"   Bổ sung {added_twins} chunk trùng nguyên văn từ tài liệu trùng lặp")
    print(f"💾 {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
