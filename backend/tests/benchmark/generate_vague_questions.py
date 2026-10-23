"""
generate_vague_questions.py — Sinh câu hỏi "MƠ HỒ" ứng viên (contextual_header_rollout_plan.md §4.4):
câu ngắn kiểu người dùng thật gõ vội, KHÔNG nêu năm / loại báo cáo / số hiệu / tên đầy đủ văn bản.

Vì sao: 131 câu benchmark được sinh "tự đủ nghĩa" (nêu rõ công ty + năm + loại báo cáo) nên khớp
mạnh với header ngữ cảnh → mức cải thiện 0.88 → 1.0 có thể bị phóng đại. Bộ này đo cả 2 phía:
header có làm câu hỏi mơ hồ tệ đi không (vd câu không nêu năm bị header "năm X" kéo lệch).

Khác generate_benchmark_questions.py:
  - 1 chunk / tài liệu mẫu (30 tài liệu của benchmark_sample_full.json), BỎ chunk đã dùng làm
    nhãn của bộ 131 câu (không trùng câu cũ).
  - Câu hỏi mơ hồ → thường NHIỀU tài liệu cùng trả lời được (vd cùng chỉ tiêu ở BCTC năm khác):
    pooling lấy ứng viên từ CẢ snapshot gốc (không header) LẪN snapshot production có header
    (benchmark_chromadb_full_prod_header, bật killswitch) để nhãn không thiên về hệ thống nào;
    judge tính đoạn nào trả lời được câu hỏi NHƯ ĐÃ ĐẶT. Khi chấm, lấy được bất kỳ tài liệu nào
    trong nhãn đều đúng (doc hit).
  - Kết quả cùng schema ứng viên → duyệt bằng export/apply/finalize với --set vague.

Dữ liệu GỬI RA NGOÀI: nội dung chunk gửi tới LLM provider (Gemini mặc định, QGEN_PROVIDER).
Chạy:
  python backend/tests/benchmark/generate_vague_questions.py --dry-run
  python backend/tests/benchmark/generate_vague_questions.py
  python backend/tests/benchmark/export_review_sheet.py --set vague
Chạy lại sẽ TIẾP TỤC từ chỗ dừng.
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

from tests.benchmark.export_review_sheet import SETS
from tests.benchmark.generate_benchmark_questions import (
    HEADER_CHARS, POOL_CHARS, LLMCallFailed, _chunk_ref, _clean_enough, build_pool_search, parse_indices,
)
from tests.benchmark.ragas_metrics import CLAUDE_JUDGE_MODEL_DEFAULT, GEMINI_JUDGE_MODEL_DEFAULT, _call_judge

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BENCH_DIR, "benchmark_chromadb_full")
HEADER_DB_PATH = os.path.join(BENCH_DIR, "benchmark_chromadb_full_prod_header")
COLLECTION_NAME = "benchmark_full"
MANIFEST_PATH = os.path.join(BENCH_DIR, "snapshot_manifest_full.json")
SAMPLE_PATH = os.path.join(BENCH_DIR, "benchmark_sample_full.json")
DATASET_PATH = SETS["full"]["output"]
OUTPUT_PATH = SETS["vague"]["candidates"]

SEED = 20261024
GENERATE_MAX_OUTPUT_TOKENS = 800
# Mẫu nhận diện mà câu mơ hồ KHÔNG được chứa — LLM lỡ nêu thì gắn cờ cho người duyệt
_SPECIFIC = re.compile(r"\b(19|20)\d{2}\b|hợp nhất|\briêng\b|số\s*\d+/|/\d{4}/|NĐ-CP|QH1\d|TT-BTC|QĐ-", re.IGNORECASE)

_GENERATE_PROMPT = """Bạn đang xây bộ câu hỏi kiểm thử "người dùng thật" cho hệ thống tìm kiếm tài liệu nội bộ doanh nghiệp (tiếng Việt). Corpus ~90 tài liệu, nhiều tài liệu RẤT GIỐNG NHAU (báo cáo tài chính cùng công ty qua nhiều năm, nghị định sửa đổi lẫn nhau, nhiều mẫu điều lệ).

TÀI LIỆU: {filename}
PHẦN ĐẦU TÀI LIỆU:
{header}

ĐOẠN NGUỒN:
{chunk}

Viết ĐÚNG 1 câu hỏi NGẮN (tối đa khoảng 15 từ) như một nhân viên gõ VỘI vào ô tìm kiếm, sao cho:
- Trả lời được từ đoạn nguồn.
- KHÔNG nêu năm, KHÔNG nêu loại báo cáo (riêng / hợp nhất / thường niên), KHÔNG nêu số hiệu văn bản, KHÔNG nêu tên đầy đủ của văn bản. Được nêu tên ngắn thường gọi của công ty / mã cổ phiếu / chủ đề (vd "Vinaconex", "MWG", "luật lao động") nếu người dùng thật sẽ nêu — không bắt buộc.
- Hỏi nội dung thực chất (quy định, điều kiện, con số, thời hạn, trách nhiệm...), không hỏi chữ ký, số trang, mục lục.
- DIỄN ĐẠT LẠI tự nhiên, không chép cụm dài hơn 4 từ liên tiếp từ đoạn nguồn (trừ tên riêng, con số).
- Câu hỏi CÓ THỂ được nhiều tài liệu tương tự cùng trả lời (vd cùng chỉ tiêu ở báo cáo năm khác) — đó là điều cần đo, KHÔNG cần tránh.

Nếu đoạn nguồn lỗi chữ, chỉ toàn bảng số không tiêu đề, hoặc không đủ nội dung → skip=true.

Trả về JSON đúng các key:
{{"skip": <true|false>, "skip_reason": "<nếu skip>", "question": "<câu hỏi>", "reference_answer": "<câu trả lời ngắn theo đoạn nguồn>", "evidence": "<trích nguyên văn ngắn từ đoạn nguồn>", "ambiguity": "<unique nếu chỉ tài liệu này trả lời được | multi nếu tài liệu tương tự khác nhiều khả năng cũng trả lời được>"}}
"""

_POOL_PROMPT = """Bạn chấm nhãn cho bộ câu hỏi tìm kiếm MƠ HỒ (câu ngắn, không nêu năm / loại báo cáo / số hiệu văn bản). Với CÂU HỎI dưới đây, xét từng ĐOẠN và cho biết đoạn nào chứa thông tin TRỰC TIẾP trả lời câu hỏi NHƯ ĐÃ ĐẶT — toàn bộ hoặc phần chính của câu trả lời.

Quan trọng: câu hỏi KHÔNG nêu năm / loại báo cáo / văn bản cụ thể, nên đoạn của tài liệu khác (năm khác, bản riêng hay hợp nhất, văn bản tương tự) VẪN TÍNH nếu nó trả lời đúng điều được hỏi cho đúng đối tượng được hỏi (đúng công ty / chủ đề), dù con số khác đáp án tham chiếu. Đoạn chỉ cùng chủ đề, nhắc khái niệm nhưng không trả lời thì KHÔNG tính.

CÂU HỎI: {question}
ĐÁP ÁN THAM CHIẾU (từ đoạn nguồn, chỉ để tham khảo): {answer}

CÁC ĐOẠN:
{candidates}

Trả về JSON: {{"relevant": [<danh sách số thứ tự các đoạn trả lời được>]}}
"""


def plan_sources(sample: dict, by_doc: dict[str, list[dict]], used_chunk_ids: set[str]) -> list[dict]:
    """1 chunk / tài liệu mẫu, sạch, chưa dùng làm nhãn ở bộ 131 câu."""
    rng = random.Random(SEED)
    tasks = []
    for doc in sample["sampled_documents"]:
        usable = [c for c in by_doc[doc["doc_id"]]
                  if c["id"] not in used_chunk_ids and _clean_enough(c["text"], doc["ext"])]
        if usable:
            tasks.append({"category": "vague", "doc": doc, "chunks": [rng.choice(usable)]})
    return tasks


def pool_relevant(question: str, answer: str, source: dict, pool: list[dict],
                  provider: str, api_key: str, model: str) -> tuple[list[dict], bool, int]:
    candidates = [source] + [c for c in pool if c["id"] != source["id"]]
    random.Random(question).shuffle(candidates)
    listing = "\n\n".join(
        f"[{i + 1}] (tài liệu: {c['meta']['filename']})\n{c['text'][:POOL_CHARS]}" for i, c in enumerate(candidates)
    )
    result = _call_judge(provider, api_key, model, _POOL_PROMPT.format(
        question=question, answer=answer, candidates=listing,
    ))
    if not isinstance(result.get("relevant"), list):
        raise LLMCallFailed(result.get("reasoning") or "pooling không trả về danh sách 'relevant'")
    picked = {candidates[idx - 1]["id"] for idx in parse_indices(result["relevant"], len(candidates))}
    others = [c for c in candidates if c["id"] in picked and c["id"] != source["id"]]
    return others, source["id"] in picked, len(candidates)


def run(dry_run: bool):
    from app.core import context_header

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        sample = json.load(f)
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    if not sample["snapshot_fingerprint"] == dataset["snapshot_fingerprint"] == manifest["fingerprint"]:
        sys.exit("❌ Sample / dataset / snapshot không cùng fingerprint.")
    used = {cid for q in dataset["questions"] for cid in q["relevant_chunk_ids"]}

    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    sampled_ids = [d["doc_id"] for d in sample["sampled_documents"]]
    data = collection.get(where={"doc_id": {"$in": sampled_ids}}, include=["documents", "metadatas"])
    by_doc: dict[str, list[dict]] = {d: [] for d in sampled_ids}
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        by_doc[meta["doc_id"]].append({"id": cid, "text": text, "meta": meta})
    for chunks in by_doc.values():
        chunks.sort(key=lambda c: (c["meta"].get("page_num") or 0, c["meta"].get("chunk_index") or 0))

    tasks = plan_sources(sample, by_doc, used)
    print(f"📋 {len(tasks)} việc sinh câu hỏi mơ hồ (bỏ {len(used)} chunk đã là nhãn của bộ 131 câu)\n")
    if dry_run:
        for t in tasks:
            preview = re.sub(r"\s+", " ", t["chunks"][0]["text"])[:120]
            print(f"  [{t['doc']['stratum']:<16}] {t['doc']['filename'][:45]:<45} | {preview}")
        print("\n(dry-run: không gửi dữ liệu ra ngoài, không ghi file)")
        return

    provider = os.getenv("QGEN_PROVIDER", os.getenv("JUDGE_PROVIDER", "gemini")).strip().lower()
    api_key = os.getenv("ANTHROPIC_API_KEY" if provider == "claude" else "GOOGLE_API_KEY")
    model = os.getenv("QGEN_MODEL", CLAUDE_JUDGE_MODEL_DEFAULT if provider == "claude" else GEMINI_JUDGE_MODEL_DEFAULT)
    if not api_key:
        sys.exit(f"❌ Thiếu API key cho provider={provider}")
    logging.basicConfig(format="   ⏳ %(message)s")
    logging.getLogger("tests.benchmark.ragas_metrics").setLevel(logging.INFO)
    print(f"🤖 provider={provider}, model={model}\n")

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            output = json.load(f)
    else:
        output = {"snapshot_fingerprint": manifest["fingerprint"],
                  "generator": {"provider": provider, "model": model, "seed": SEED, "kind": "vague"},
                  "candidates": [], "skipped": []}
    done = {c["task_key"] for c in output["candidates"]} | {s["task_key"] for s in output["skipped"]}

    def save():
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    # Pool từ CẢ 2 hệ thống: bật killswitch trước khi dựng BM25 (snapshot gốc không có metadata
    # header nên không bị ảnh hưởng; snapshot có header thì BM25 tokenize "header + nội dung")
    context_header.set_enabled(True)
    search_base = build_pool_search(collection)
    search_header = build_pool_search(chromadb.PersistentClient(path=HEADER_DB_PATH).get_collection(COLLECTION_NAME))
    equivalents = {d["doc_id"]: d["equivalent_filenames"] for d in sample["sampled_documents"]}
    failed = 0

    for i, task in enumerate(tasks, 1):
        source = task["chunks"][0]
        key = f"vague:{source['id']}"
        if key in done:
            continue
        doc = task["doc"]
        print(f"[{i}/{len(tasks)}] {doc['stratum']} — {doc['filename']}")
        gen = _call_judge(provider, api_key, model, _GENERATE_PROMPT.format(
            filename=doc["filename"], header=by_doc[doc["doc_id"]][0]["text"][:HEADER_CHARS], chunk=source["text"],
        ), max_output_tokens=GENERATE_MAX_OUTPUT_TOKENS)
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
        pool = {c["id"]: c for c in search_base(question) + search_header(question)}
        try:
            pooled, confirmed, pool_size = pool_relevant(
                question, gen.get("reference_answer", ""), source, list(pool.values()), provider, api_key, model)
        except LLMCallFailed as exc:
            failed += 1
            print(f"   ❌ lỗi pooling, để lần chạy sau làm lại: {exc}")
            continue
        relevant = [{**_chunk_ref(source), "via": "source"}] + [{**_chunk_ref(c), "via": "pooled"} for c in pooled]
        relevant_files = sorted({r["filename"] for r in relevant} | set(equivalents[doc["doc_id"]]))
        flags = []
        if not confirmed:
            flags.append("source_not_confirmed")
        if any(r["filename"] != doc["filename"] for r in relevant):
            flags.append("answer_also_in_other_document")
        if _SPECIFIC.search(question):
            flags.append("mentions_year_type_or_number")

        output["candidates"].append({
            "id": f"v{len(output['candidates']) + 1:03d}",
            "task_key": key,
            "question": question,
            "reference_answer": gen.get("reference_answer"),
            "evidence": gen.get("evidence"),
            # LLM đoán unique|multi — chỉ tham khảo, KHÔNG làm cờ: lần chạy đầu nó đoán "multi" cho 30/30 câu
            "llm_ambiguity": str(gen.get("ambiguity", "")).strip().lower() or None,
            "category": "vague",
            "stratum": doc["stratum"],
            "source_chunks": [_chunk_ref(source)],
            "relevant_chunks": relevant,
            "expected_sources": [{"filename": f, "page_num": None} for f in relevant_files],
            "pooling": {"pool_size": pool_size, "source_confirmed": confirmed, "added_chunks": len(pooled),
                        "systems": ["baseline_no_header", "prod_header"]},
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
    run(dry_run=parser.parse_args().dry_run)
