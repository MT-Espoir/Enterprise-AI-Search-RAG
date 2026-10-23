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

Chạy: python backend/tests/benchmark/build_benchmark_db_full.py [--force]
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


def build(force: bool = False):
    source_path = (os.getenv("CHROMA_PERSIST_PATH") or "").strip() or SOURCE_PATH_DEFAULT
    source_collection_name = os.getenv("CHROMA_COLLECTION_NAME", SOURCE_COLLECTION_DEFAULT)

    if os.path.exists(DB_PATH):
        if not force:
            print(f"❌ Snapshot đã tồn tại tại {DB_PATH}.")
            print("   Snapshot là corpus ĐÓNG BĂNG mà dataset dựa vào — chạy lại với --force")
            print("   chỉ khi chủ động muốn tạo corpus mới (dataset cũ sẽ không còn khớp fingerprint).")
            sys.exit(1)
        shutil.rmtree(DB_PATH)

    print(f"📥 Nguồn: {source_path} (collection={source_collection_name})")
    source = chromadb.PersistentClient(path=source_path).get_collection(source_collection_name)
    total = source.count()
    print(f"   {total} chunks\n")

    target = chromadb.PersistentClient(path=DB_PATH).create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    all_ids: list[str] = []
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
        "documents": sorted(
            (
                {"doc_id": doc_id, "filename": doc_filenames[doc_id], "chunk_count": count}
                for doc_id, count in per_doc.items()
            ),
            key=lambda d: d["filename"] or "",
        ),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Snapshot: {copied} chunks / {len(per_doc)} tài liệu tại {DB_PATH}")
    print(f"   Fingerprint: {manifest['fingerprint']}")
    print(f"   Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Xóa snapshot cũ và chụp lại")
    build(force=parser.parse_args().force)
