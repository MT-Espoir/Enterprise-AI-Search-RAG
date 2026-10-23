"""
sample_benchmark_docs.py — Lấy mẫu PHÂN TẦNG ~30 tài liệu từ snapshot
benchmark_chromadb_full/ làm nguồn sinh câu hỏi cho benchmark tổng quát.

Vì sao phân tầng theo NHÓM NỘI DUNG chứ không theo đuôi file hay ngẫu nhiên thuần:
  - Ngẫu nhiên thuần trên 89 tài liệu → PDF chiếm 73%, Excel/DOCX gần như vắng mặt.
  - Đuôi file không phản ánh độ khó retrieval: 23 báo cáo tài chính PDF (nhiều năm,
    cùng công ty, rất giống nhau) khó phân biệt hơn hẳn 28 văn bản luật khác chủ đề.
    Mỗi nhóm có kiểu nhầm lẫn riêng nên cần có mặt đủ trong benchmark.

Phân bổ: tỷ lệ theo kích thước nhóm, tối thiểu MIN_PER_STRATUM mỗi nhóm.
Tài liệu KHÔNG đủ điều kiện làm NGUỒN câu hỏi (vẫn nằm trong corpus làm nhiễu):
  - quá ít chunk (< MIN_CHUNKS), bản sao trùng nội dung của tài liệu khác,
  - hỏng mã hóa (font VNI/TCVN3 chưa chuyển Unicode) — sinh câu hỏi từ đó vô nghĩa.

Nhóm tài liệu tương đương (equivalence_groups): các tài liệu trùng ≥ DUP_OVERLAP
số chunk. Câu hỏi sinh từ 1 tài liệu trong nhóm thì MỌI tài liệu trong nhóm đều
được tính là đúng — tránh chấm sai oan retriever.

Chạy: python backend/tests/benchmark/sample_benchmark_docs.py
"""
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

import chromadb

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BENCH_DIR, "benchmark_chromadb_full")
COLLECTION_NAME = "benchmark_full"
MANIFEST_PATH = os.path.join(BENCH_DIR, "snapshot_manifest_full.json")
OUTPUT_PATH = os.path.join(BENCH_DIR, "benchmark_sample_full.json")

SEED = 20261022
TARGET_DOCS = 30
MIN_PER_STRATUM = 2
MIN_CHUNKS = 5
DUP_OVERLAP = 0.5
# Ký tự Latin KHÔNG có trong tiếng Việt — dấu hiệu font VNI/TCVN3 hiển thị sai
LEGACY_ENCODING_CHARS = re.compile(r"[ÏØÄÖÑÛËÜÆïøäöñûëüæ]")
LEGACY_CHUNK_THRESHOLD = 3
LEGACY_DOC_RATIO = 0.3

CORPORATE = {
    "dieu-le-cong-ty-tnhh-1-thanh-vien-so-huu-ca-nhan_.docx",
    "dieu-le-cong-ty-tnhh-1-thanh-vien-so-huu-to-chuc.docx",
    "mau-dieu-le-cong-ty-co-phan-mau-1.docx",
    "mau-dieu-le-cong-ty-co-phan-mau-2.docx",
    "mau-dieu-le-tnhh-2-thanh-vien-tro-len.docx",
    "quyet-dinh-ban-hanh-quy-che-tien-luong_1005174201.docx",
    "quyet-dinh-ban-hanh-quy-che_1005174151.docx",
    "quyet-dinh-ban-hanh-quy-che_1005174151_1_-_Copy.docx",
    "3._Mau_hop_dong_QLDA.pdf",
    "6._Mau_hop_dong_thi_cong.pdf",
    "Du-thao-Quy-che-Quan-tri-2018_TV_1.pdf",
    "Du_thao_thoa_uoc_lao_dong_tap_the.pdf",
    "4431_qd_2023_22a46.pdf",
    "356-Q-UBND.pdf",
    "5_QUY_TRINH_TONG_HOP_BAO_CAO.pdf",
    "Signed.1.-Quy-trinh-quan-ly-de-tai-cac-cap.pdf",
    "4.-1.005370.-K-NCCB.pdf",
    "WFYJswVux0tHYmVJMUC_LUC_ISO.pdf",
}
OTHER = {
    "1811.12808v3.pdf",
    "2312.10997v5.pdf",
    "System_Design_Deep_Feature_Analysis.md",
    "ktmt_OnTap.pdf",
    "THMLHK_223.pdf",
    "Ban_tin_KHCNDN_so_6.2022.pdf",
}


def stratum_of(filename: str) -> str:
    lower = filename.lower()
    if lower.startswith("simplize_") and lower.endswith(".xlsx"):
        return "financial_xlsx"
    if "bctc" in lower or lower.startswith("bc-quan-tri"):
        return "financial_report"
    if filename in CORPORATE:
        return "corporate"
    if filename in OTHER:
        return "other"
    return "legal"


def _norm_hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def equivalence_groups(chunk_hashes: dict[str, set[str]]) -> list[list[str]]:
    """Gom tài liệu trùng nội dung (≥ DUP_OVERLAP chunk của tài liệu nhỏ hơn) bằng union-find."""
    parent = {f: f for f in chunk_hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    names = sorted(chunk_hashes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            smaller = min(len(chunk_hashes[a]), len(chunk_hashes[b]))
            if smaller and len(chunk_hashes[a] & chunk_hashes[b]) / smaller >= DUP_OVERLAP:
                parent[find(a)] = find(b)

    groups = defaultdict(list)
    for f in names:
        groups[find(f)].append(f)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Phân bổ tỷ lệ (largest remainder), tối thiểu MIN_PER_STRATUM, không vượt kích thước nhóm."""
    pool = sum(sizes.values())
    quota = {s: max(MIN_PER_STRATUM, total * n / pool) for s, n in sizes.items()}
    alloc = {s: min(sizes[s], int(q)) for s, q in quota.items()}
    remainders = sorted(sizes, key=lambda s: quota[s] - int(quota[s]), reverse=True)
    for s in remainders:
        if sum(alloc.values()) >= total:
            break
        if alloc[s] < sizes[s]:
            alloc[s] += 1
    return alloc


def run():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])

    chunk_hashes: dict[str, set[str]] = defaultdict(set)
    legacy_chunks: dict[str, int] = defaultdict(int)
    for text, meta in zip(data["documents"], data["metadatas"]):
        filename = meta["filename"]
        chunk_hashes[filename].add(_norm_hash(text))
        if len(LEGACY_ENCODING_CHARS.findall(text)) >= LEGACY_CHUNK_THRESHOLD:
            legacy_chunks[filename] += 1

    groups = equivalence_groups(chunk_hashes)
    # Trong mỗi nhóm trùng, chỉ tài liệu đầu tiên (theo tên) đủ điều kiện làm nguồn
    duplicate_of = {f: g[0] for g in groups for f in g[1:]}

    docs = []
    for d in manifest["documents"]:
        filename = d["filename"]
        legacy_ratio = legacy_chunks[filename] / d["chunk_count"]
        reasons = []
        if d["chunk_count"] < MIN_CHUNKS:
            reasons.append(f"chỉ {d['chunk_count']} chunk")
        if filename in duplicate_of:
            reasons.append(f"trùng nội dung {duplicate_of[filename]}")
        if legacy_ratio >= LEGACY_DOC_RATIO:
            reasons.append(f"hỏng mã hóa font cũ ({legacy_ratio:.0%} chunk)")
        docs.append({
            **d,
            "stratum": stratum_of(filename),
            "ext": os.path.splitext(filename)[1].lower(),
            "eligible": not reasons,
            "excluded_reason": "; ".join(reasons) or None,
        })

    eligible_by_stratum = defaultdict(list)
    for d in docs:
        if d["eligible"]:
            eligible_by_stratum[d["stratum"]].append(d)

    sizes = {s: len(v) for s, v in eligible_by_stratum.items()}
    alloc = allocate(sizes, TARGET_DOCS)

    rng = random.Random(SEED)
    sampled = []
    for stratum in sorted(alloc):
        pool = sorted(eligible_by_stratum[stratum], key=lambda d: d["filename"])
        sampled.extend(rng.sample(pool, alloc[stratum]))

    group_of = {f: g for g in groups for f in g}
    output = {
        "seed": SEED,
        "snapshot_fingerprint": manifest["fingerprint"],
        "allocation": alloc,
        "stratum_sizes_eligible": sizes,
        "equivalence_groups": groups,
        "sampled_documents": [
            {
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "stratum": d["stratum"],
                "ext": d["ext"],
                "chunk_count": d["chunk_count"],
                "equivalent_filenames": [f for f in group_of.get(d["filename"], []) if f != d["filename"]],
            }
            for d in sorted(sampled, key=lambda d: (d["stratum"], d["filename"]))
        ],
        "excluded_documents": [
            {"filename": d["filename"], "stratum": d["stratum"], "reason": d["excluded_reason"]}
            for d in docs if not d["eligible"]
        ],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Nhóm tài liệu trùng nội dung: {groups}\n")
    print("Không đủ điều kiện làm nguồn:")
    for d in output["excluded_documents"]:
        print(f"  - [{d['stratum']}] {d['filename']}: {d['reason']}")
    print(f"\nPhân bổ (đủ điều kiện → lấy): " + ", ".join(f"{s} {sizes[s]}→{alloc[s]}" for s in sorted(alloc)))
    print(f"\nĐã chọn {len(sampled)} tài liệu (seed={SEED}):")
    for d in output["sampled_documents"]:
        print(f"  [{d['stratum']:<16}] {d['chunk_count']:>4} chunk  {d['filename']}")
    print(f"\n💾 {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
