"""
generate_benchmark_questions.py — Sinh câu hỏi ỨNG VIÊN cho benchmark tổng quát từ
30 tài liệu đã lấy mẫu (benchmark_sample_full.json), kèm nhãn đúng mức CHUNK.

Kết quả là ỨNG VIÊN chờ người duyệt (review.status = "pending"), KHÔNG dùng thẳng
làm benchmark — câu hỏi do LLM sinh luôn cần người kiểm tra trước khi tin.

Quy trình cho mỗi câu hỏi:
  1. Chọn chunk nguồn (seed cố định, rải đều theo vị trí trong tài liệu, bỏ chunk
     quá ngắn/hỏng mã hóa). Loại "multi_chunk": 2 chunk xa nhau trong cùng tài liệu.
  2. LLM sinh 1 câu hỏi DIỄN ĐẠT LẠI (tránh chép nguyên văn — làm BM25 đẹp giả tạo),
     TỰ ĐỦ NGHĨA (nêu rõ tên văn bản/công ty/năm vì corpus có nhiều tài liệu rất giống
     nhau), hoặc SKIP nếu chunk không đủ nội dung.
  3. POOLING: lấy top chunk từ vector + BM25 trên snapshot, LLM đánh dấu chunk nào
     CŨNG trả lời được câu hỏi → thêm vào relevant_chunks. Tránh chấm sai oan khi văn
     bản luật sửa đổi/báo cáo tài chính năm sau chứa cùng thông tin. Chunk nguồn được
     trộn lẫn vào danh sách như một ĐỐI CHỨNG: nếu LLM không xác nhận chunk nguồn trả lời
     được → gắn cờ source_not_confirmed để người duyệt xem kỹ.

Dữ liệu GỬI RA NGOÀI: nội dung chunk được gửi tới LLM provider (Gemini/Claude theo
QGEN_PROVIDER). Dùng --dry-run để xem chunk nào sẽ được chọn mà không gửi gì.

Chạy:
  python backend/tests/benchmark/generate_benchmark_questions.py --dry-run
  python backend/tests/benchmark/generate_benchmark_questions.py [--limit-docs N]
Chạy lại sẽ TIẾP TỤC từ chỗ dừng (bỏ qua chunk nguồn đã xử lý).
"""
import argparse
import json
import logging
import os
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb

from tests.benchmark.ragas_metrics import (
    CLAUDE_JUDGE_MODEL_DEFAULT,
    GEMINI_JUDGE_MODEL_DEFAULT,
    _call_judge,
)
from tests.benchmark.sample_benchmark_docs import LEGACY_CHUNK_THRESHOLD, LEGACY_ENCODING_CHARS

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BENCH_DIR, "benchmark_chromadb_full")
COLLECTION_NAME = "benchmark_full"
MANIFEST_PATH = os.path.join(BENCH_DIR, "snapshot_manifest_full.json")
SAMPLE_PATH = os.path.join(BENCH_DIR, "benchmark_sample_full.json")
OUTPUT_PATH = os.path.join(BENCH_DIR, "evaluation_dataset_full_candidates.json")

SEED = 20261022
SINGLE_PER_DOC = 4
MULTI_CHUNK_TOTAL = 20
MULTI_CHUNK_STRATA = ("legal", "corporate", "financial_report")
MIN_CHUNK_CHARS = 400
POOL_PER_SYSTEM = 10
POOL_CHARS = 1000
HEADER_CHARS = 400
GENERATE_MAX_OUTPUT_TOKENS = 1000

_GENERATE_PROMPT = """Bạn đang xây bộ benchmark cho hệ thống tìm kiếm tài liệu nội bộ doanh nghiệp (tiếng Việt). Corpus có ~90 tài liệu, trong đó nhiều tài liệu RẤT GIỐNG NHAU (vd báo cáo tài chính cùng công ty qua nhiều năm, các nghị định sửa đổi lẫn nhau, nhiều mẫu điều lệ công ty).

TÀI LIỆU: {filename}
PHẦN ĐẦU TÀI LIỆU (để biết tên văn bản/công ty/năm):
{header}

ĐOẠN NGUỒN:
{chunks}

Viết ĐÚNG 1 câu hỏi mà một nhân viên thật sẽ gõ vào ô tìm kiếm, sao cho:
- Trả lời được CHỈ từ {scope}. {multi_rule}
- TỰ ĐỦ NGHĨA khi đứng một mình: nêu rõ tên văn bản/số hiệu, hoặc tên công ty + năm + loại báo cáo, đủ để phân biệt với các tài liệu tương tự. KHÔNG viết "văn bản này", "tài liệu trên".
- DIỄN ĐẠT LẠI bằng từ ngữ tự nhiên, KHÔNG chép nguyên cụm dài hơn 4 từ liên tiếp từ đoạn nguồn (trừ tên riêng, số hiệu, con số).
- Hỏi về nội dung thực chất (quy định, điều kiện, con số, thời hạn, trách nhiệm...), không hỏi về chữ ký, số trang, mục lục.

Nếu đoạn nguồn bị lỗi chữ, chỉ toàn bảng số không có tiêu đề, hoặc không đủ nội dung để đặt câu hỏi có ý nghĩa → trả về skip=true.

Trả về JSON đúng các key:
{{"skip": <true|false>, "skip_reason": "<nếu skip>", "question": "<câu hỏi>", "reference_answer": "<câu trả lời ngắn, dựa trên đoạn nguồn>", "evidence": "<trích nguyên văn ngắn từ đoạn nguồn chứng minh câu trả lời>"}}
"""

_POOL_PROMPT = """Bạn chấm nhãn cho benchmark tìm kiếm tài liệu. Với CÂU HỎI và ĐÁP ÁN THAM CHIẾU dưới đây, xét từng ĐOẠN và cho biết đoạn nào chứa thông tin TRỰC TIẾP trả lời câu hỏi — toàn bộ đáp án, hoặc một phần chính của đáp án (vd câu hỏi liệt kê nhiều ý mà đoạn chứa vài ý trong đó). Đoạn chỉ cùng chủ đề, nhắc tới khái niệm nhưng không chứa nội dung của đáp án thì KHÔNG tính. Chú ý: đoạn của tài liệu khác (năm khác, văn bản khác) chỉ tính nếu nó thực sự trả lời đúng câu hỏi như đã đặt (đúng công ty, đúng năm, đúng văn bản được hỏi).

CÂU HỎI: {question}
ĐÁP ÁN THAM CHIẾU: {answer}

CÁC ĐOẠN:
{candidates}

Trả về JSON: {{"relevant": [<danh sách số thứ tự các đoạn trả lời được>]}}
"""


def _clean_enough(text: str, ext: str) -> bool:
    if len(text) < MIN_CHUNK_CHARS:
        return False
    if len(LEGACY_ENCODING_CHARS.findall(text)) >= LEGACY_CHUNK_THRESHOLD:
        return False
    if ext != ".xlsx":
        letters = sum(ch.isalpha() for ch in text)
        if letters / len(text) < 0.4:
            return False
    return True


def _spread_pick(rng: random.Random, chunks: list[dict], n: int) -> list[dict]:
    """Chọn n chunk rải đều: chia tài liệu thành n đoạn liên tiếp, mỗi đoạn lấy ngẫu nhiên 1."""
    if len(chunks) <= n:
        return list(chunks)
    bucket = len(chunks) / n
    return [rng.choice(chunks[int(i * bucket):int((i + 1) * bucket)]) for i in range(n)]


def _chunk_ref(chunk: dict) -> dict:
    return {
        "chunk_id": chunk["id"],
        "doc_id": chunk["meta"]["doc_id"],
        "filename": chunk["meta"]["filename"],
        "page_num": chunk["meta"].get("page_num"),
    }


def plan_sources(sample: dict, by_doc: dict[str, list[dict]]) -> list[dict]:
    """Danh sách việc cần sinh: mỗi phần tử là {category, stratum, doc, chunks:[...]}."""
    rng = random.Random(SEED)
    tasks = []
    for doc in sample["sampled_documents"]:
        usable = [c for c in by_doc[doc["doc_id"]] if _clean_enough(c["text"], doc["ext"])]
        for chunk in _spread_pick(rng, usable, SINGLE_PER_DOC):
            tasks.append({"category": "single_chunk", "doc": doc, "chunks": [chunk]})

    multi_docs = [d for d in sample["sampled_documents"] if d["stratum"] in MULTI_CHUNK_STRATA]
    for i in range(MULTI_CHUNK_TOTAL):
        doc = multi_docs[i % len(multi_docs)]
        usable = [c for c in by_doc[doc["doc_id"]] if _clean_enough(c["text"], doc["ext"])]
        if len(usable) < 4:
            continue
        half = len(usable) // 2
        tasks.append({
            "category": "multi_chunk",
            "doc": doc,
            "chunks": [rng.choice(usable[:half]), rng.choice(usable[half:])],
        })
    return tasks


def _task_key(task: dict) -> str:
    return task["category"] + ":" + "+".join(c["id"] for c in task["chunks"])


def generate_question(task: dict, header: str, provider: str, api_key: str, model: str) -> dict:
    multi = task["category"] == "multi_chunk"
    chunks_text = "\n\n".join(
        f"[Đoạn {i + 1}]\n{c['text']}" for i, c in enumerate(task["chunks"])
    )
    prompt = _GENERATE_PROMPT.format(
        filename=task["doc"]["filename"],
        header=header,
        chunks=chunks_text,
        scope="CẢ HAI đoạn nguồn kết hợp" if multi else "đoạn nguồn",
        multi_rule=(
            "Câu hỏi PHẢI cần thông tin từ cả Đoạn 1 và Đoạn 2 — không trả lời đủ được nếu chỉ đọc 1 đoạn. "
            "Nếu 2 đoạn không liên quan đủ để ghép thành 1 câu hỏi tự nhiên → skip=true."
            if multi else ""
        ),
    )
    # 500 token (mặc định của judge) từng cắt cụt JSON khi reference_answer + evidence dài
    return _call_judge(provider, api_key, model, prompt, max_output_tokens=GENERATE_MAX_OUTPUT_TOKENS)


def parse_indices(raw: list, upper: int) -> set[int]:
    """Số thứ tự đoạn (1-based) từ output LLM. Gemini có lúc trả chuỗi ("9") thay vì số (9)
    — đã gặp thật: bỏ qua chuỗi khiến chunk nguồn bị coi là "không xác nhận" sai."""
    indices = set()
    for item in raw:
        try:
            idx = int(float(str(item).strip().strip("[]")))
        except ValueError:
            continue
        if 1 <= idx <= upper:
            indices.add(idx)
    return indices


class LLMCallFailed(Exception):
    """Lời gọi LLM lỗi (503 quá tải, hết retry...) — KHÔNG được lưu thành kết quả, để lần
    chạy sau làm lại. Khác với skip (LLM trả lời hợp lệ rằng chunk không dùng được)."""


def pool_relevant(question: str, answer: str, source_chunks: list[dict], pool: list[dict],
                  provider: str, api_key: str, model: str) -> tuple[list[dict], bool, int]:
    """Trả về (chunk khác được xác nhận trả lời được, chunk nguồn có được xác nhận không, cỡ pool)."""
    source_ids = {c["id"] for c in source_chunks}
    candidates = list(source_chunks) + [c for c in pool if c["id"] not in source_ids]
    random.Random(question).shuffle(candidates)
    listing = "\n\n".join(
        f"[{i + 1}] (tài liệu: {c['meta']['filename']})\n{c['text'][:POOL_CHARS]}"
        for i, c in enumerate(candidates)
    )
    result = _call_judge(provider, api_key, model, _POOL_PROMPT.format(
        question=question, answer=answer, candidates=listing,
    ))
    if not isinstance(result.get("relevant"), list):
        raise LLMCallFailed(result.get("reasoning") or "pooling không trả về danh sách 'relevant'")
    picked = {candidates[idx - 1]["id"] for idx in parse_indices(result["relevant"], len(candidates))}
    confirmed = all(cid in picked for cid in source_ids)
    others = [c for c in candidates if c["id"] in picked and c["id"] not in source_ids]
    return others, confirmed, len(candidates)


def build_pool_search(collection):
    """Vector (BGE-M3, như production) + BM25 trên snapshot — import nặng chỉ khi cần."""
    from app.core.retrieval import BM25Index, Retriever
    from app.ingestion.embedder.local_embedder import LocalEmbedder
    from app.vectorstore.operations import VectorStoreOps

    class SnapshotOps(VectorStoreOps):
        def __init__(self, coll):
            self.collection = coll

    ops = SnapshotOps(collection)
    vector = Retriever(ops=ops, embedder=LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL")),
                       top_k=POOL_PER_SYSTEM)
    bm25 = BM25Index()
    bm25.refresh(ops)

    def search(question: str) -> list[dict]:
        found = {}
        for c in vector.retrieve(question, acl_bypass=True):
            found[c.chunk_id] = {"id": c.chunk_id, "text": c.text, "meta": {
                "doc_id": c.doc_id, "filename": c.filename, "page_num": c.page}}
        for hit in bm25.search(question, top_k=POOL_PER_SYSTEM, acl_bypass=True):
            found.setdefault(hit["id"], {"id": hit["id"], "text": hit["text"], "meta": hit["metadata"]})
        return list(found.values())

    return search


def run(dry_run: bool, limit_docs: int | None):
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        sample = json.load(f)
    if sample["snapshot_fingerprint"] != manifest["fingerprint"]:
        sys.exit("❌ benchmark_sample_full.json không khớp snapshot hiện tại — chạy lại sample_benchmark_docs.py.")

    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    sampled_ids = [d["doc_id"] for d in sample["sampled_documents"]]
    data = collection.get(where={"doc_id": {"$in": sampled_ids}}, include=["documents", "metadatas"])
    by_doc: dict[str, list[dict]] = {d: [] for d in sampled_ids}
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        by_doc[meta["doc_id"]].append({"id": cid, "text": text, "meta": meta})
    for chunks in by_doc.values():
        chunks.sort(key=lambda c: (c["meta"].get("page_num") or 0, c["meta"].get("chunk_index") or 0))

    # Lập kế hoạch trên ĐỦ mẫu rồi mới lọc: chạy thử --limit-docs là tập con đúng của lần
    # chạy đầy đủ (chạy tiếp sẽ bỏ qua phần đã làm, không sinh câu lệch kế hoạch)
    tasks = plan_sources(sample, by_doc)
    if limit_docs:
        allowed = {d["doc_id"] for d in sample["sampled_documents"][:limit_docs]}
        tasks = [t for t in tasks if t["doc"]["doc_id"] in allowed]

    print(f"📋 {len(tasks)} việc sinh câu hỏi "
          f"({sum(t['category'] == 'single_chunk' for t in tasks)} single_chunk, "
          f"{sum(t['category'] == 'multi_chunk' for t in tasks)} multi_chunk)\n")

    if dry_run:
        for t in tasks:
            preview = re.sub(r"\s+", " ", t["chunks"][0]["text"])[:120]
            print(f"  [{t['category']:<12}] {t['doc']['filename'][:45]:<45} | {preview}")
        print("\n(dry-run: không gửi dữ liệu ra ngoài, không ghi file)")
        return

    provider = os.getenv("QGEN_PROVIDER", os.getenv("JUDGE_PROVIDER", "gemini")).strip().lower()
    if provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("QGEN_MODEL", CLAUDE_JUDGE_MODEL_DEFAULT)
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        model = os.getenv("QGEN_MODEL", GEMINI_JUDGE_MODEL_DEFAULT)
    if not api_key:
        sys.exit(f"❌ Thiếu API key cho provider={provider}")
    # Hiện log chờ rate limit (15 RPM / 250k TPM free tier) — các thư viện khác giữ WARNING
    logging.basicConfig(format="   ⏳ %(message)s")
    logging.getLogger("tests.benchmark.ragas_metrics").setLevel(logging.INFO)
    print(f"🤖 Sinh câu hỏi + pooling bằng provider={provider}, model={model}\n")

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            output = json.load(f)
    else:
        output = {
            "snapshot_fingerprint": manifest["fingerprint"],
            "generator": {"provider": provider, "model": model, "seed": SEED},
            "candidates": [],
            "skipped": [],
        }
    done = {c["task_key"] for c in output["candidates"]} | {s["task_key"] for s in output["skipped"]}

    def save():
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    search = build_pool_search(collection)
    equivalents = {d["doc_id"]: d["equivalent_filenames"] for d in sample["sampled_documents"]}
    failed = 0

    for i, task in enumerate(tasks, 1):
        key = _task_key(task)
        if key in done:
            continue
        doc = task["doc"]
        header = by_doc[doc["doc_id"]][0]["text"][:HEADER_CHARS]
        print(f"[{i}/{len(tasks)}] {task['category']} — {doc['filename']}")

        gen = generate_question(task, header, provider, api_key, model)
        if "skip" not in gen and "question" not in gen:
            failed += 1
            print(f"   ❌ lỗi gọi LLM, để lần chạy sau làm lại: {gen.get('reasoning')}")
            continue
        if gen.get("skip") or not gen.get("question"):
            reason = gen.get("skip_reason") or "không có câu hỏi"
            print(f"   ⏭  skip: {reason}")
            output["skipped"].append({"task_key": key, "filename": doc["filename"], "reason": reason})
            save()
            continue

        question = gen["question"].strip()
        try:
            pooled, source_confirmed, pool_size = pool_relevant(
                question, gen.get("reference_answer", ""), task["chunks"], search(question),
                provider, api_key, model,
            )
        except LLMCallFailed as exc:
            failed += 1
            print(f"   ❌ lỗi pooling, để lần chạy sau làm lại: {exc}")
            continue
        sources = [_chunk_ref(c) for c in task["chunks"]]
        relevant = [{**r, "via": "source"} for r in sources] + [{**_chunk_ref(c), "via": "pooled"} for c in pooled]
        relevant_files = sorted({r["filename"] for r in relevant} | set(equivalents[doc["doc_id"]]))
        flags = []
        if not source_confirmed:
            flags.append("source_not_confirmed")
        if any(r["filename"] != doc["filename"] for r in relevant):
            flags.append("answer_also_in_other_document")

        output["candidates"].append({
            "id": f"q{len(output['candidates']) + 1:03d}",
            "task_key": key,
            "question": question,
            "reference_answer": gen.get("reference_answer"),
            "evidence": gen.get("evidence"),
            "category": task["category"],
            "stratum": doc["stratum"],
            "source_chunks": sources,
            "relevant_chunks": relevant,
            "expected_sources": [{"filename": f, "page_num": None} for f in relevant_files],
            "pooling": {"pool_size": pool_size, "source_confirmed": source_confirmed,
                        "added_chunks": len(pooled)},
            "flags": flags,
            "review": {"status": "pending", "note": ""},
        })
        save()
        print(f"   ✅ {question}")
        print(f"      relevant: {len(relevant)} chunk / {len(relevant_files)} tài liệu"
              + (f"  ⚠ {', '.join(flags)}" if flags else ""))

    print(f"\n💾 {len(output['candidates'])} ứng viên, {len(output['skipped'])} skip → {OUTPUT_PATH}")
    if failed:
        print(f"⚠️  {failed} việc lỗi gọi LLM (chưa lưu) — chạy lại lệnh để làm tiếp phần còn thiếu.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in chunk sẽ được chọn, không gọi LLM")
    parser.add_argument("--limit-docs", type=int, default=None, help="Chỉ xử lý N tài liệu đầu (chạy thử)")
    args = parser.parse_args()
    run(dry_run=args.dry_run, limit_docs=args.limit_docs)
