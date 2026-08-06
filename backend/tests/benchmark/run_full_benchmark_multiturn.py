"""
run_full_benchmark_multiturn.py — Phase 3(b): đo lợi ích thật của Query Rewrite
(QueryProcessor rewrite path dựa trên history) bằng cách so sánh 2 điều kiện
trên CÙNG một tập câu hỏi tỉnh lược/đại từ tham chiếu nhiều lượt hội thoại:

  - Điều kiện A "WITH_REWRITE" (khớp hành vi production, xem rag_pipeline.py):
    gọi query_processor.process(target_question, history=history) → nếu
    plan.rewrite=True, dùng plan.rewritten_query cho nhánh vector retrieval.
  - Điều kiện B "NO_REWRITE" (baseline, giả lập KHÔNG có QueryProcessor rewrite):
    retrieval thẳng bằng target_question gốc, vector_query=None, bỏ qua history.

Generator ở CẢ HAI điều kiện đều nhận `history` giống nhau (đúng hành vi
production — generator luôn thấy lịch sử hội thoại bất kể rewrite hay không) —
để cô lập chính xác: chênh lệch retrieval/generation chỉ đến từ việc
vector_query có được viết lại hay không, không phải từ việc generator có
ngữ cảnh hay không.

Dataset: evaluation_dataset_multiturn.json (15 hội thoại, dùng lại
benchmark_chromadb_legal đã build sẵn ở Phase 3(a) — KHÔNG build DB riêng).

Yêu cầu: đã chạy build_benchmark_db_legal.py, Ollama đang chạy, GOOGLE_API_KEY
trong .env (chỉ dùng cho judge).

Chạy: python backend/tests/benchmark/run_full_benchmark_multiturn.py
"""
import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb
import requests

from app.ingestion.embedder.local_embedder import LocalEmbedder
from app.vectorstore.operations import VectorStoreOps
from app.core.retrieval import BM25Index
from app.core.retrieval import build_retriever
from app.core.query import QueryProcessor
from app.core.ranking import Reranker
from app.core.generation import LocalGenerator

from tests.benchmark.metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_recall_at_k,
    calculate_precision_at_k,
)
from tests.benchmark.ragas_metrics import (
    score_faithfulness,
    score_answer_relevancy,
    is_refusal_answer,
    GEMINI_JUDGE_MODEL_DEFAULT,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "benchmark_chromadb_legal")
COLLECTION_NAME = "benchmark_legal"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
LOCAL_GENERATOR_MODEL_DEFAULT = "qwen2.5:3b-instruct"

CONDITIONS = ["with_rewrite", "no_rewrite"]


class MockVectorStoreOps(VectorStoreOps):
    def __init__(self, collection):
        self.collection = collection


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 3) if clean else None


def _run_one_condition(condition, target_question, history, expected, query_processor, retriever, reranker, generator):
    """Chạy 1 điều kiện (with_rewrite / no_rewrite) cho 1 câu hỏi target, trả về record."""
    rewrite_used = False
    rewritten_query = None

    if condition == "with_rewrite":
        plan = query_processor.process(target_question, history=history)
        rewrite_used = plan.rewrite
        rewritten_query = plan.rewritten_query if plan.rewrite else None
        vector_query = rewritten_query
    else:
        vector_query = None

    candidates = retriever.retrieve(target_question, vector_query=vector_query)

    if not candidates:
        answer = "Tôi không tìm thấy tài liệu nào liên quan đến câu hỏi của bạn."
        top_chunks = []
    else:
        top_chunks = reranker.rerank(target_question, candidates)
        gen_result = generator.generate(target_question, top_chunks, history=history)
        answer = gen_result.answer

    h_rate = calculate_hit_rate(candidates, expected)
    mrr = calculate_mrr(candidates, expected)
    r3 = calculate_recall_at_k(candidates, expected, 3)
    p3 = calculate_precision_at_k(candidates, expected, 3)
    r5 = calculate_recall_at_k(candidates, expected, min(5, len(candidates) or 1))
    p5 = calculate_precision_at_k(candidates, expected, min(5, len(candidates) or 1))
    hit_rate_post_rerank = calculate_hit_rate(top_chunks, expected)

    return {
        "condition": condition,
        "rewrite_used": rewrite_used,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "top_chunk_texts": [c.text for c in top_chunks] if candidates else [],
        "retrieval_metrics": {
            "hit_rate": h_rate,
            "mrr": mrr,
            "recall@3": r3,
            "precision@3": p3,
            "recall@5": r5,
            "precision@5": p5,
            "hit_rate_post_rerank@3": hit_rate_post_rerank,
        },
    }


def run(limit: int = None):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Thiếu GOOGLE_API_KEY trong .env (chỉ cần cho judge, không cần cho generator/embedding)")
        sys.exit(1)

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL_DEFAULT)
    local_llm_model = os.getenv("LOCAL_LLM_MODEL", LOCAL_GENERATOR_MODEL_DEFAULT)
    judge_model = os.getenv("GEMINI_JUDGE_MODEL", GEMINI_JUDGE_MODEL_DEFAULT)

    try:
        requests.get(f"{ollama_base_url.rstrip('/')}/api/tags", timeout=5).raise_for_status()
    except Exception:
        print(f"❌ Không kết nối được Ollama tại {ollama_base_url}. Chạy `ollama serve` trước.")
        sys.exit(1)

    if not os.path.exists(DB_PATH):
        print(f"❌ Không tìm thấy benchmark DB tại {DB_PATH}.")
        print("   Chạy trước: python backend/tests/benchmark/build_benchmark_db_legal.py")
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"❌ Collection '{COLLECTION_NAME}' không tồn tại. Chạy build_benchmark_db_legal.py trước.")
        sys.exit(1)

    vector_ops = MockVectorStoreOps(collection)
    embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL"))

    bm25_index = BM25Index()
    bm25_index.refresh(vector_ops)
    retriever = build_retriever("hybrid", ops=vector_ops, embedder=embedder, bm25_index=bm25_index, top_k=10)

    query_processor = QueryProcessor(
        base_url=ollama_base_url, model_name=local_llm_model,
        default_strategy="hybrid", default_top_k=10, slm_enabled=True,
    )
    reranker = Reranker.get_instance(top_k=3)
    generator = LocalGenerator(base_url=ollama_base_url, model_name=local_llm_model)

    dataset_path = os.path.join(os.path.dirname(__file__), "evaluation_dataset_multiturn.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if limit:
        dataset = dataset[:limit]

    print(f"🚀 Phase 3(b) — Benchmark multi-turn (đo lợi ích Query Rewrite) trên {len(dataset)} hội thoại "
          f"× {len(CONDITIONS)} điều kiện...\n")

    detailed_results = []

    for idx, item in enumerate(dataset):
        conv_id = item["conversation_id"]
        history = item["history"]
        target_question = item["target_question"]
        category = item["category"]
        expected = item["expected_sources"]
        is_negative = category == "negative_control"

        print(f"\n[{idx + 1}/{len(dataset)}] ({category}) {conv_id}")
        print(f"    history: {' | '.join(h['content'][:60] for h in history)}")
        print(f"    target : {target_question}")

        for condition in CONDITIONS:
            cond_result = _run_one_condition(
                condition, target_question, history, expected,
                query_processor, retriever, reranker, generator,
            )

            refusal = is_refusal_answer(cond_result["answer"])
            cond_result["is_refusal"] = refusal

            if is_negative:
                score = 1.0 if refusal else 0.0
                reasoning = (
                    "Rule-based: model từ chối đúng cách cho câu hỏi ngoài phạm vi tài liệu."
                    if refusal else
                    "Rule-based: model KHÔNG từ chối dù câu hỏi ngoài phạm vi tài liệu — coi là bịa đặt."
                )
                cond_result["ragas"] = {
                    "faithfulness": score, "faithfulness_reasoning": reasoning,
                    "answer_relevancy": score, "answer_relevancy_reasoning": reasoning,
                    "rule_override": True,
                }
            else:
                faithfulness = score_faithfulness(
                    target_question, cond_result["answer"], cond_result["top_chunk_texts"],
                    api_key=api_key, model_name=judge_model,
                )
                relevancy = score_answer_relevancy(
                    target_question, cond_result["answer"], api_key=api_key,
                    is_negative_expected=False, model_name=judge_model,
                )
                cond_result["ragas"] = {
                    "faithfulness": faithfulness.get("score"),
                    "faithfulness_reasoning": faithfulness.get("reasoning"),
                    "answer_relevancy": relevancy.get("score"),
                    "answer_relevancy_reasoning": relevancy.get("reasoning"),
                    "rule_override": False,
                }

            rm = cond_result["retrieval_metrics"]
            answer_preview = cond_result["answer"][:200].replace("\n", " ")
            print(f"    [{condition:^13}] rewrite={cond_result['rewrite_used']} "
                  f"rewritten_query={cond_result['rewritten_query']!r}")
            print(f"      Hit={rm['hit_rate']} MRR={rm['mrr']:.2f} "
                  f"Faithfulness={cond_result['ragas']['faithfulness']} "
                  f"Relevancy={cond_result['ragas']['answer_relevancy']}")
            print(f"      A: {answer_preview}{'...' if len(cond_result['answer']) > 200 else ''}")

            detailed_results.append({
                "conversation_id": conv_id,
                "category": category,
                "target_question": target_question,
                **cond_result,
            })

    # ── Tổng kết: so sánh 2 điều kiện, tổng thể + theo category ────────────
    summary = {"total_conversations": len(dataset), "by_condition": {}, "by_category_and_condition": {}}

    for condition in CONDITIONS:
        cond_records = [r for r in detailed_results if r["condition"] == condition]
        non_negative = [r for r in cond_records if r["category"] != "negative_control"]
        summary["by_condition"][condition] = {
            "hit_rate": _mean([r["retrieval_metrics"]["hit_rate"] for r in non_negative]),
            "mrr": _mean([r["retrieval_metrics"]["mrr"] for r in non_negative]),
            "recall@3": _mean([r["retrieval_metrics"]["recall@3"] for r in non_negative]),
            "recall@5": _mean([r["retrieval_metrics"]["recall@5"] for r in non_negative]),
            "hit_rate_post_rerank@3": _mean([r["retrieval_metrics"]["hit_rate_post_rerank@3"] for r in non_negative]),
            "faithfulness": _mean([r["ragas"]["faithfulness"] for r in cond_records]),
            "answer_relevancy": _mean([r["ragas"]["answer_relevancy"] for r in cond_records]),
            "refusal_rate": round(sum(r["is_refusal"] for r in cond_records) / len(cond_records), 3),
            "rewrite_triggered_count": sum(1 for r in cond_records if r["rewrite_used"]),
        }

    for cat in sorted(set(r["category"] for r in detailed_results)):
        summary["by_category_and_condition"][cat] = {}
        for condition in CONDITIONS:
            cat_cond_records = [r for r in detailed_results if r["category"] == cat and r["condition"] == condition]
            non_neg = cat_cond_records if cat != "negative_control" else []
            summary["by_category_and_condition"][cat][condition] = {
                "count": len(cat_cond_records),
                "hit_rate": _mean([r["retrieval_metrics"]["hit_rate"] for r in non_neg]) if non_neg else None,
                "faithfulness": _mean([r["ragas"]["faithfulness"] for r in cat_cond_records]),
                "answer_relevancy": _mean([r["ragas"]["answer_relevancy"] for r in cat_cond_records]),
                "refusal_rate": round(sum(r["is_refusal"] for r in cat_cond_records) / len(cat_cond_records), 3),
            }

    print("\n" + "=" * 60)
    print("📊 BÁO CÁO TỔNG KẾT — Phase 3(b) Multi-turn Query Rewrite")
    print("=" * 60)
    print(f"Tổng số hội thoại: {summary['total_conversations']}\n")
    for condition in CONDITIONS:
        s = summary["by_condition"][condition]
        print(f"-- {condition} (rewrite triggered {s['rewrite_triggered_count']}/{len(dataset)}) --")
        for k, v in s.items():
            if k == "rewrite_triggered_count":
                continue
            print(f"  {k:<28}: {v}")
        print()

    print("-- Theo category (hit_rate: with_rewrite vs no_rewrite) --")
    for cat, by_cond in summary["by_category_and_condition"].items():
        wr = by_cond["with_rewrite"]
        nr = by_cond["no_rewrite"]
        print(f"  {cat:<22} (n={wr['count']:<2}) hit_rate: {wr['hit_rate']} vs {nr['hit_rate']}  "
              f"| faithfulness: {wr['faithfulness']} vs {nr['faithfulness']}")
    print("=" * 60)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(REPORTS_DIR, f"report_multiturn_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": detailed_results}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Báo cáo chi tiết đã lưu: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3(b) multi-turn Query Rewrite benchmark.")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N hội thoại đầu (smoke test).")
    args = parser.parse_args()
    run(limit=args.limit)
