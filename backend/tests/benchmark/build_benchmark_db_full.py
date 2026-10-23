"""
build_benchmark_db_full.py — Chụp SNAPSHOT toàn bộ collection production
(CHROMA_PERSIST_PATH, collection enterprise_documents — 89 tài liệu tại thời điểm
tạo) sang một ChromaDB benchmark RIÊNG (benchmark_chromadb_full/).

Vì sao snapshot thay vì chạy benchmark thẳng trên DB production:
  - Production thay đổi mỗi lần user upload/xóa → cùng 1 dataset cho ra số khác nhau
    giữa 2 lần chạy, không so sánh trước/sau được. Snapshot đóng băng corpus.
  - Tránh đọc SQLite của Chroma trong lúc Flask server đang ghi.

Vì sao copy qua Chroma API thay vì copy thư mục: copy file SQLite + segment HNSW khi
server đang chạy có thể ra bản không nhất quán. Đọc qua API (chỉ đọc, không sửa
production) rồi add lại embedding sẵn có — KHÔNG embed lại, giữ nguyên vector production.

Ghi kèm snapshot_manifest_full.json: thời điểm, nguồn, số chunk theo từng tài liệu,
fingerprint (SHA-256 của danh sách chunk id) — dataset sinh từ snapshot nào thì
benchmark kiểm tra đúng fingerprint đó trước khi chạy.

Manifest ghi thêm `context_headers` (số chunk/tài liệu có metadata context_header + hash bộ header):
fingerprint chỉ băm chunk_id nên KHÔNG phân biệt được embedding có header hay chưa.

Chạy: python backend/tests/benchmark/build_benchmark_db_full.py [--force]
      python backend/tests/benchmark/build_benchmark_db_full.py --out benchmark_chromadb_full_prod_header
        (--out: chụp sang thư mục KHÁC + manifest riêng snapshot_manifest_<tên>.json — giữ nguyên
         snapshot gốc làm baseline; dataset vẫn dùng được nếu fingerprint trùng)
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BENCH_DIR, "benchmark_chromadb_full")
COLLECTION_NAME = "benchmark_full"
MANIFEST_PATH = os.path.join(BENCH_DIR, "snapshot_manifest_full.json")

SOURCE_PATH_DEFAULT = os.path.abspath(os.path.join(BENCH_DIR, "..", "..", "..", "chromadb_data"))
SOURCE_COLLECTION_DEFAULT = "enterprise_documents"
BATCH_SIZE = 500


def corpus_fingerprint(chunk_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(chunk_ids)).encode("utf-8")).hexdigest()


def context_headers_summary(metadatas: list[dict]) -> dict:
    """Cùng công thức hash với run_benchmark_full.describe_context_headers / poc_contextual_headers_all."""
    header_of = {m["doc_id"]: m["context_header"].strip() for m in metadatas if (m.get("context_header") or "").strip()}
    payload = "\n".join(f"{doc_id}\x1f{header_of[doc_id]}" for doc_id in sorted(header_of))
    return {
        "chunks_with_header": sum(bool((m.get("context_header") or "").strip()) for m in metadatas),
        "documents_with_header": len(header_of),
        "headers_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest() if header_of else None,
    }


def manifest_path_for(db_path: str) -> str:
    name = os.path.basename(os.path.normpath(db_path))
    return MANIFEST_PATH if name == os.path.basename(DB_PATH) else os.path.join(BENCH_DIR, f"snapshot_manifest_{name}.json")


def build(force: bool = False, db_path: str = DB_PATH):
    manifest_path = manifest_path_for(db_path)
    source_path = (os.getenv("CHROMA_PERSIST_PATH") or "").strip() or SOURCE_PATH_DEFAULT
    source_collection_name = os.getenv("CHROMA_COLLECTION_NAME", SOURCE_COLLECTION_DEFAULT)

    if os.path.exists(db_path):
        if not force:
            print(f"❌ Snapshot đã tồn tại tại {db_path}.")
            print("   Snapshot là corpus ĐÓNG BĂNG mà dataset dựa vào — chạy lại với --force")
            print("   chỉ khi chủ động muốn tạo corpus mới (dataset cũ sẽ không còn khớp fingerprint).")
            sys.exit(1)
        shutil.rmtree(db_path)

    print(f"📥 Nguồn: {source_path} (collection={source_collection_name})")
    source = chromadb.PersistentClient(path=source_path).get_collection(source_collection_name)
    total = source.count()
    print(f"   {total} chunks\n")

    target = chromadb.PersistentClient(path=db_path).create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    # Cùng ef_search với nguồn — ef_search khác thì kết quả ANN khác production (đo 2026-10-24)
    source_hnsw = (source.configuration or {}).get("hnsw") or {}
    if source_hnsw.get("ef_search"):
        target.modify(configuration={"hnsw": {"ef_search": source_hnsw["ef_search"]}})

    all_ids: list[str] = []
    all_metas: list[dict] = []
    per_doc: Counter = Counter()
    doc_filenames: dict[str, str] = {}

    for offset in range(0, total, BATCH_SIZE):
        batch = source.get(
            limit=BATCH_SIZE, offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        target.add(
            ids=batch["ids"],
            embeddings=batch["embeddings"],
            documents=batch["documents"],
            metadatas=batch["metadatas"],
        )
        all_ids.extend(batch["ids"])
        all_metas.extend(batch["metadatas"])
        for meta in batch["metadatas"]:
            doc_id = meta.get("doc_id")
            per_doc[doc_id] += 1
            doc_filenames.setdefault(doc_id, meta.get("filename"))
        print(f"   💾 {min(offset + BATCH_SIZE, total)}/{total}")

    copied = target.count()
    assert copied == total == len(set(all_ids)), (
        f"❌ Số chunk không khớp: nguồn={total}, đích={copied}, id duy nhất={len(set(all_ids))}"
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_path": source_path,
        "source_collection": source_collection_name,
        "collection": COLLECTION_NAME,
        "total_chunks": copied,
        "total_documents": len(per_doc),
        "fingerprint": corpus_fingerprint(all_ids),
        "context_headers": context_headers_summary(all_metas),
        "hnsw": {k: source_hnsw.get(k) for k in ("space", "ef_construction", "ef_search", "max_neighbors")},
        "documents": sorted(
            (
                {"doc_id": doc_id, "filename": doc_filenames[doc_id], "chunk_count": count}
                for doc_id, count in per_doc.items()
            ),
            key=lambda d: d["filename"] or "",
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Snapshot: {copied} chunks / {len(per_doc)} tài liệu tại {db_path}")
    print(f"   Fingerprint: {manifest['fingerprint']}")
    print(f"   Header ngữ cảnh: {manifest['context_headers']}")
    print(f"   Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Xóa snapshot cũ và chụp lại")
    parser.add_argument("--out", default=None,
                        help="Tên thư mục snapshot (trong tests/benchmark/) — mặc định benchmark_chromadb_full")
    args = parser.parse_args()
    build(force=args.force, db_path=os.path.join(BENCH_DIR, args.out) if args.out else DB_PATH)
