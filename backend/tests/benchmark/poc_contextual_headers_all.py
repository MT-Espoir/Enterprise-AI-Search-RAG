"""
poc_contextual_headers_all.py — S5 (contextual_header_rollout_plan.md §4.1): PoC header ngữ
cảnh cho CẢ 89 tài liệu, chạy trên BẢN SAO snapshot (benchmark_chromadb_full_ctx_all/),
KHÔNG đụng production lẫn snapshot gốc benchmark_chromadb_full/ (giữ làm baseline).

Khác PoC 21 BCTC (poc_contextual_headers.py — header nằm luôn trong document text):
bản này lưu ĐÚNG như production sẽ làm (S1, app/core/context_header.py):
  - documents giữ NỘI DUNG GỐC, header chỉ nằm ở metadata["context_header"];
  - embedding = build_context_text(header, text);
  - BM25/Reranker thấy header qua context_text_for() khi benchmark chạy với
    --context-header (bật killswitch) → kiểm tra luôn đường đi dây của S1.

Header = header user đã duyệt ở S4, đọc CHỈ-ĐỌC từ MongoDB rag_db, chụp lại vào
poc_contextual_headers_all.json (gitignore) để lần chạy sau tái lập được.

Tiết kiệm thời gian: chunk BCTC đã embed ở PoC 21 tài liệu (benchmark_chromadb_full_ctx/)
được DÙNG LẠI vector nếu header trùng chuỗi (cùng công thức "header\\nnội dung").
Chạy lại an toàn: chunk đã có đúng header trong metadata thì bỏ qua.

Chạy: python backend/tests/benchmark/poc_contextual_headers_all.py [--refresh-headers]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb

from app.core.context_header import build_context_text

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DB = os.path.join(BENCH_DIR, "benchmark_chromadb_full")
REUSE_DB = os.path.join(BENCH_DIR, "benchmark_chromadb_full_ctx")
TARGET_DB = os.path.join(BENCH_DIR, "benchmark_chromadb_full_ctx_all")
COLLECTION_NAME = "benchmark_full"
MANIFEST_PATH = os.path.join(BENCH_DIR, "snapshot_manifest_full.json")
HEADERS_PATH = os.path.join(BENCH_DIR, "poc_contextual_headers_all.json")
BATCH_SIZE = 32


def headers_hash(header_of: dict[str, str]) -> str:
    """Băm (doc_id, header) — ghi vào report để biết benchmark chạy với bộ header nào."""
    payload = "\n".join(f"{doc_id}\x1f{header_of[doc_id]}" for doc_id in sorted(header_of))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_headers_from_mongo(manifest: dict) -> dict:
    from bson import ObjectId
    from pymongo import MongoClient

    db = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/rag_db")).get_default_database()
    wanted = {d["doc_id"]: d["filename"] for d in manifest["documents"]}
    docs = {str(d["_id"]): d for d in db.documents.find(
        {"_id": {"$in": [ObjectId(i) for i in wanted]}}, {"context_header": 1, "filename": 1})}
    missing = [wanted[i] for i in wanted if not (docs.get(i) or {}).get("context_header")]
    if missing:
        sys.exit(f"❌ {len(missing)} tài liệu trong snapshot chưa có header đã duyệt: {missing[:5]}")
    header_of = {i: docs[i]["context_header"].strip() for i in wanted}
    config = {
        "status": "ĐÃ DUYỆT (S4, đọc từ rag_db)",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_db": db.name,
        "snapshot_fingerprint": manifest["fingerprint"],
        "headers_hash": headers_hash(header_of),
        "headers": [{"doc_id": i, "filename": wanted[i], "header": header_of[i]} for i in wanted],
    }
    with open(HEADERS_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"📝 Đã chụp {len(header_of)} header từ {db.name} → {HEADERS_PATH}")
    return config


def load_headers(manifest: dict, refresh: bool) -> dict[str, str]:
    if refresh or not os.path.exists(HEADERS_PATH):
        config = snapshot_headers_from_mongo(manifest)
    else:
        with open(HEADERS_PATH, encoding="utf-8") as f:
            config = json.load(f)
    if config["snapshot_fingerprint"] != manifest["fingerprint"]:
        sys.exit("❌ Bộ header được chụp cho snapshot khác — chạy lại với --refresh-headers.")
    return {h["doc_id"]: h["header"] for h in config["headers"]}


def reusable_embeddings(header_of: dict[str, str]) -> dict[str, list[float]]:
    """chunk_id → embedding từ PoC 21 BCTC nếu header PoC trùng chuỗi header hiện tại."""
    if not os.path.exists(REUSE_DB):
        return {}
    data = chromadb.PersistentClient(path=REUSE_DB).get_collection(COLLECTION_NAME).get(
        where={"doc_id": {"$in": list(header_of)}}, include=["metadatas", "embeddings"])
    return {
        cid: list(emb) for cid, meta, emb in zip(data["ids"], data["metadatas"], data["embeddings"])
        if meta.get("context_header") and meta["context_header"].strip() == header_of.get(meta["doc_id"])
    }


def run(refresh_headers: bool):
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    header_of = load_headers(manifest, refresh_headers)

    if not os.path.exists(TARGET_DB):
        print(f"📋 Sao chép snapshot gốc → {TARGET_DB}")
        shutil.copytree(SOURCE_DB, TARGET_DB)
    collection = chromadb.PersistentClient(path=TARGET_DB).get_collection(COLLECTION_NAME)

    data = collection.get(include=["documents", "metadatas"])
    unknown = {m["doc_id"] for m in data["metadatas"]} - set(header_of)
    if unknown:
        sys.exit(f"❌ Chunk thuộc tài liệu không có header: {sorted(unknown)[:5]}")
    todo = [
        (cid, text, meta) for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
        if meta.get("context_header") != header_of[meta["doc_id"]]
    ]
    print(f"🧩 {len(data['ids'])} chunk / {len(header_of)} tài liệu — cần cập nhật: {len(todo)}")

    reuse = reusable_embeddings(header_of)
    reused = [t for t in todo if t[0] in reuse]
    to_embed = [t for t in todo if t[0] not in reuse]
    for start in range(0, len(reused), 256):
        batch = reused[start:start + 256]
        collection.update(ids=[cid for cid, _, _ in batch], embeddings=[reuse[cid] for cid, _, _ in batch],
                          metadatas=[{**meta, "context_header": header_of[meta["doc_id"]]} for _, _, meta in batch])
    if reused:
        print(f"♻️  Dùng lại vector PoC 21 BCTC: {len(reused)} chunk")

    if to_embed:
        from app.ingestion.embedder.local_embedder import LocalEmbedder
        embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL"))
        # Chunk ngắn trước → batch đều độ dài, ít padding; tiến độ ước lượng thận trọng hơn
        to_embed.sort(key=lambda t: len(t[1]))
        t0 = time.perf_counter()
        for start in range(0, len(to_embed), BATCH_SIZE):
            batch = to_embed[start:start + BATCH_SIZE]
            texts = [build_context_text(header_of[meta["doc_id"]], text) for _, text, meta in batch]
            # documents KHÔNG đổi (nội dung gốc) — chỉ embedding + metadata, đúng quy ước S1
            collection.update(ids=[cid for cid, _, _ in batch], embeddings=embedder.embed_batch(texts),
                              metadatas=[{**meta, "context_header": header_of[meta["doc_id"]]} for _, _, meta in batch])
            done = start + len(batch)
            rate = done / (time.perf_counter() - t0)
            if done == len(to_embed) or (start // BATCH_SIZE) % 10 == 0:
                print(f"   💾 {done}/{len(to_embed)} ({rate:.2f} chunk/s, còn ~{(len(to_embed) - done) / rate / 60:.0f} phút)",
                      flush=True)

    metas = collection.get(include=["metadatas"])["metadatas"]
    ok = sum(m.get("context_header") == header_of[m["doc_id"]] for m in metas)
    print(f"✅ Xong: {ok}/{len(metas)} chunk có đúng header trong {TARGET_DB}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-headers", action="store_true", help="Chụp lại header từ MongoDB rag_db")
    run(refresh_headers=parser.parse_args().refresh_headers)
