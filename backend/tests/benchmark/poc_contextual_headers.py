"""
poc_contextual_headers.py — PoC "header ngữ cảnh tài liệu" cho 21 báo cáo tài chính,
chạy trên BẢN SAO snapshot (benchmark_chromadb_full_ctx/), KHÔNG đụng production lẫn
snapshot gốc benchmark_chromadb_full/ (giữ làm baseline so sánh).

Vấn đề cần kiểm chứng (benchmark dev 2026-10-22): nhóm financial_report chỉ đạt rerank doc
hit@3 0.577 — chunk BCTC là mảnh trang, chỉ 36% chứa năm / 19% chứa "riêng|hợp nhất" /
20% chứa tên công ty → retriever lấy nhầm năm hoặc nhầm riêng/hợp nhất.

Cách làm: mỗi chunk của 21 BCTC → "<header>\\n<nội dung cũ>", embed lại bằng đúng embedder
production (BGE-M3), GIỮ NGUYÊN chunk_id để nhãn benchmark (relevant_chunk_ids) vẫn dùng
được. Header nằm trong document text nên embedding, BM25 VÀ Reranker đều thấy.
Header đã được user duyệt: poc_contextual_headers.json.

Chạy lại an toàn: chunk đã có metadata context_header thì bỏ qua (không ghép header 2 lần).

Chạy: python backend/tests/benchmark/poc_contextual_headers.py
"""
import json
import os
import shutil
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb

from app.ingestion.embedder.local_embedder import LocalEmbedder

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DB = os.path.join(BENCH_DIR, "benchmark_chromadb_full")
TARGET_DB = os.path.join(BENCH_DIR, "benchmark_chromadb_full_ctx")
COLLECTION_NAME = "benchmark_full"
HEADERS_PATH = os.path.join(BENCH_DIR, "poc_contextual_headers.json")
BATCH_SIZE = 64


def run():
    with open(HEADERS_PATH, encoding="utf-8") as f:
        config = json.load(f)
    if not config["status"].startswith("ĐÃ DUYỆT"):
        sys.exit("❌ Header chưa được user duyệt (poc_contextual_headers.json).")
    header_of = {h["filename"]: h["header"] for h in config["headers"]}

    if not os.path.exists(TARGET_DB):
        print(f"📋 Sao chép snapshot → {TARGET_DB}")
        shutil.copytree(SOURCE_DB, TARGET_DB)
    collection = chromadb.PersistentClient(path=TARGET_DB).get_collection(COLLECTION_NAME)

    data = collection.get(where={"filename": {"$in": list(header_of)}}, include=["documents", "metadatas"])
    todo = [
        (cid, text, meta) for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
        if not meta.get("context_header")
    ]
    found_files = {m["filename"] for m in data["metadatas"]}
    missing = set(header_of) - found_files
    if missing:
        sys.exit(f"❌ Không tìm thấy trong snapshot: {sorted(missing)}")
    print(f"🧩 {len(data['ids'])} chunk thuộc {len(found_files)} BCTC — cần embed lại: {len(todo)}")

    embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL"))
    t0 = time.perf_counter()
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        ids = [cid for cid, _, _ in batch]
        texts = [f"{header_of[meta['filename']]}\n{text}" for _, text, meta in batch]
        metas = [{**meta, "context_header": header_of[meta["filename"]]} for _, _, meta in batch]
        collection.update(ids=ids, embeddings=embedder.embed_batch(texts), documents=texts, metadatas=metas)
        done = start + len(batch)
        rate = done / (time.perf_counter() - t0)
        print(f"   💾 {done}/{len(todo)} ({rate:.1f} chunk/s, còn ~{(len(todo) - done) / rate / 60:.0f} phút)")

    # Đếm ở Python: where {"$ne": ""} của Chroma khớp cả chunk KHÔNG có trường này
    metas = collection.get(include=["metadatas"])["metadatas"]
    with_header = sum(bool(m.get("context_header")) for m in metas)
    print(f"✅ Xong: {with_header} chunk có header / {len(metas)} chunk trong {TARGET_DB}")


if __name__ == "__main__":
    run()
