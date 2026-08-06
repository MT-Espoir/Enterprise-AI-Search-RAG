"""
run_full_benchmark_legal.py — Biến thể run_full_benchmark.py cho corpus luật
(benchmark_chromadb_legal/, evaluation_dataset_legal.json), TÁCH BIỆT hoàn
toàn khỏi benchmark IoT gốc (không đụng benchmark_chromadb/, evaluation_dataset.json,
report_*.json cũ) — xem rag_core_quality_roadmap.md mục 6c/6d.

Khác biệt so với bản gốc: dataset có 2 câu "summary" cho 2 tài liệu KHÁC NHAU
(bản gốc chỉ có 1 tài liệu benchmark_doc_001 cho mọi câu summary) — nên cần map
filename -> doc_id theo từng câu hỏi thay vì dùng 1 hằng số DOC_ID cố định.

Yêu cầu chạy build_benchmark_db_legal.py trước, Ollama đang chạy, GOOGLE_API_KEY
trong .env (chỉ dùng cho judge).

Chạy: python backend/tests/benchmark/run_full_benchmark_legal.py
"""
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
from app.core.rag_pipeline import RAGPipeline

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
    CLAUDE_JUDGE_MODEL_DEFAULT,
)
from tests.benchmark.latency import StageTimer, summarize_latency, warm_up_pipeline
from tests.benchmark.build_benchmark_db_legal import TEST_FILES

DB_PATH = os.path.join(os.path.dirname(__file__), "benchmark_chromadb_legal")
COLLECTION_NAME = "benchmark_legal"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
LOCAL_GENERATOR_MODEL_DEFAULT = "qwen2.5:3b-instruct"

# filename (như trong evaluation_dataset_legal.json) -> doc_id (như trong build_benchmark_db_legal.py)
FILENAME_TO_DOC_ID = {f["filename"]: f["doc_id"] for f in TEST_FILES}


class MockVectorStoreOps(VectorStoreOps):
    def __init__(self, collection):
        self.collection = collection


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 3) if clean else None


def run():
    judge_provider = os.getenv("JUDGE_PROVIDER", "gemini").strip().lower()
    if judge_provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ Thiếu ANTHROPIC_API_KEY trong .env (cần cho judge Claude — xem JUDGE_PROVIDER=claude)")
            sys.exit(1)
        judge_model = os.getenv("ANTHROPIC_JUDGE_MODEL", CLAUDE_JUDGE_MODEL_DEFAULT)
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ Thiếu GOOGLE_API_KEY trong .env (chỉ cần cho judge, không cần cho generator/embedding)")
            sys.exit(1)
        judge_model = os.getenv("GEMINI_JUDGE_MODEL", GEMINI_JUDGE_MODEL_DEFAULT)
    print(f"⚖️  Judge: provider={judge_provider}, model={judge_model}\n")

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL_DEFAULT)
    local_llm_model = os.getenv("LOCAL_LLM_MODEL", LOCAL_GENERATOR_MODEL_DEFAULT)

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

    strategy = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
    bm25_index = None
    if strategy == "hybrid":
        bm25_index = BM25Index()
        rebuilt = bm25_index.refresh(vector_ops)
        chunk_count = len(vector_ops.get_all_chunks())
        print(f"📚 BM25 index: {'rebuilt' if rebuilt else 'skipped (unchanged)'}, {chunk_count} chunks.\n")

    retriever = build_retriever(strategy, ops=vector_ops, embedder=embedder, bm25_index=bm25_index, top_k=10)

    slm_enabled = os.getenv("QUERY_PROCESSOR_SLM_ENABLED", "true").lower() == "true"
    query_processor = QueryProcessor(
        base_url=ollama_base_url, model_name=local_llm_model,
        default_strategy=strategy, default_top_k=10, slm_enabled=slm_enabled,
    )
    print(f"🧠 QueryProcessor: SLM {'BẬT' if slm_enabled else 'TẮT (chỉ fast-path)'}\n")

    reranker = Reranker.get_instance(top_k=3)
    generator = LocalGenerator(base_url=ollama_base_url, model_name=local_llm_model)
    rag = RAGPipeline(retriever=retriever, reranker=reranker, generator=generator, query_processor=query_processor)

    dataset_path = os.path.join(os.path.dirname(__file__), "evaluation_dataset_legal.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("🔥 Warm-up (nạp model, KHÔNG tính vào số đo)...")
    warm_secs = warm_up_pipeline(
        query_processor, retriever, reranker, generator,
        warm_question="Nội dung chính của văn bản này là gì?",
    )
    print(f"   Warm-up xong trong {warm_secs}s.\n")

    print(f"🚀 Chạy full benchmark (corpus luật) trên {len(dataset)} câu hỏi...\n")

    detailed_results = []

    for idx, item in enumerate(dataset):
        question = item["question"]
        expected = item["expected_sources"]
        category = item.get("category", "factual")
        is_negative = category == "negative"
        print(f"[{idx + 1}/{len(dataset)}] ({category}) Q: {question}")

        record = {"question": question, "category": category}
        timer = StageTimer()

        if category == "summary":
            # Khác bản gốc: doc_id lấy theo filename của CÂU HỎI NÀY, không phải 1 hằng số cố định
            doc_id = FILENAME_TO_DOC_ID[expected[0]["filename"]]
            with timer.stage("generation_ms"):
                result = rag.run(question, doc_id=doc_id)
            answer = result["answer"]
            raw_summary_chunks = retriever.ops.get_first_chunks_of_doc(doc_id, limit=10)
            top_chunk_texts = [c["text"] for c in raw_summary_chunks]
            record.update({"answer": answer, "retrieval_metrics": None})
        else:
            with timer.stage("preprocessing_ms"):
                plan = query_processor.process(question)
            vector_query = plan.rewritten_query if plan.rewrite else None
            with timer.stage("retrieval_ms"):
                candidates = retriever.retrieve(question, vector_query=vector_query)

            if not candidates:
                answer = "Tôi không tìm thấy tài liệu nào liên quan đến câu hỏi của bạn."
                top_chunks = []
            else:
                with timer.stage("rerank_ms"):
                    top_chunks = reranker.rerank(question, candidates)
                with timer.stage("generation_ms"):
                    gen_result = generator.generate(question, top_chunks)
                answer = gen_result.answer

            top_chunk_texts = [c.text for c in top_chunks] if candidates else []

            h_rate = calculate_hit_rate(candidates, expected)
            mrr = calculate_mrr(candidates, expected)
            r3 = calculate_recall_at_k(candidates, expected, 3)
            p3 = calculate_precision_at_k(candidates, expected, 3)
            r5 = calculate_recall_at_k(candidates, expected, min(5, len(candidates) or 1))
            p5 = calculate_precision_at_k(candidates, expected, min(5, len(candidates) or 1))
            hit_rate_post_rerank = calculate_hit_rate(top_chunks, expected)

            record.update(
                {
                    "answer": answer,
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
            )

        record["latency_ms"] = timer.as_record()

        refusal = is_refusal_answer(answer)
        record["is_refusal"] = refusal

        if is_negative:
            score = 1.0 if refusal else 0.0
            reasoning = (
                "Rule-based: model từ chối đúng cách cho câu hỏi ngoài phạm vi tài liệu."
                if refusal
                else "Rule-based: model KHÔNG từ chối dù câu hỏi ngoài phạm vi tài liệu — coi là bịa đặt (hallucination)."
            )
            record["ragas"] = {
                "faithfulness": score,
                "faithfulness_reasoning": reasoning,
                "answer_relevancy": score,
                "answer_relevancy_reasoning": reasoning,
                "rule_override": True,
            }
        else:
            faithfulness = score_faithfulness(
                question, answer, top_chunk_texts, api_key=api_key,
                model_name=judge_model, provider=judge_provider,
            )
            relevancy = score_answer_relevancy(
                question, answer, api_key=api_key, is_negative_expected=False,
                model_name=judge_model, provider=judge_provider,
            )
            record["ragas"] = {
                "faithfulness": faithfulness.get("score"),
                "faithfulness_reasoning": faithfulness.get("reasoning"),
                "answer_relevancy": relevancy.get("score"),
                "answer_relevancy_reasoning": relevancy.get("reasoning"),
                "rule_override": False,
            }

        detailed_results.append(record)

        answer_preview = answer[:300].replace("\n", " ")
        print(f"    A: {answer_preview}{'...' if len(answer) > 300 else ''}")

        rm = record["retrieval_metrics"]
        rm_str = f"Hit={rm['hit_rate']} MRR={rm['mrr']:.2f}" if rm else "Hit=N/A (summary)"
        faith_score = record["ragas"]["faithfulness"]
        rel_score = record["ragas"]["answer_relevancy"]
        e2e = record["latency_ms"]["end_to_end_ms"]
        print(f"    {rm_str} | Faithfulness={faith_score} | Relevancy={rel_score} | {e2e}ms")

        if faith_score is None or faith_score < 0.5:
            print(f"       ⚠️  Faithfulness reasoning: {record['ragas']['faithfulness_reasoning']}")
        if rel_score is None or rel_score < 0.5:
            print(f"       ⚠️  Relevancy reasoning: {record['ragas']['answer_relevancy_reasoning']}")

    qa_records = [r for r in detailed_results if r["retrieval_metrics"] is not None]
    non_negative_qa = [r for r in qa_records if r["category"] != "negative"]
    refused_despite_hit = [
        r for r in non_negative_qa if r["is_refusal"] and r["retrieval_metrics"]["hit_rate"] == 1
    ]

    summary = {
        "total_questions": len(dataset),
        "by_category": {},
        "retrieval": {
            "hit_rate": _mean([r["retrieval_metrics"]["hit_rate"] for r in qa_records]),
            "mrr": _mean([r["retrieval_metrics"]["mrr"] for r in qa_records]),
            "recall@3": _mean([r["retrieval_metrics"]["recall@3"] for r in qa_records]),
            "precision@3": _mean([r["retrieval_metrics"]["precision@3"] for r in qa_records]),
            "recall@5": _mean([r["retrieval_metrics"]["recall@5"] for r in qa_records]),
            "precision@5": _mean([r["retrieval_metrics"]["precision@5"] for r in qa_records]),
            "hit_rate_post_rerank@3": _mean(
                [r["retrieval_metrics"]["hit_rate_post_rerank@3"] for r in qa_records]
            ),
        },
        "generation": {
            "faithfulness": _mean([r["ragas"]["faithfulness"] for r in detailed_results]),
            "answer_relevancy": _mean([r["ragas"]["answer_relevancy"] for r in detailed_results]),
        },
        "latency": summarize_latency([r["latency_ms"] for r in detailed_results]),
        "refusal": {
            "refusal_rate_overall": round(
                sum(r["is_refusal"] for r in detailed_results) / len(detailed_results), 3
            ),
            "refusal_count_despite_hit": len(refused_despite_hit),
            "refusal_count_despite_hit_total_non_negative_qa": len(non_negative_qa),
            "note": (
                "Số câu bị model từ chối trả lời dù retriever đã lấy đúng tài liệu (hit_rate=1) "
                "— chỉ dấu bottleneck nằm ở generator (đọc-hiểu context), không phải retrieval."
            ),
        },
    }

    for cat in sorted(set(r["category"] for r in detailed_results)):
        cat_records = [r for r in detailed_results if r["category"] == cat]
        cat_e2e = summarize_latency([r["latency_ms"] for r in cat_records])["end_to_end_ms"]
        summary["by_category"][cat] = {
            "count": len(cat_records),
            "faithfulness": _mean([r["ragas"]["faithfulness"] for r in cat_records]),
            "answer_relevancy": _mean([r["ragas"]["answer_relevancy"] for r in cat_records]),
            "refusal_rate": round(sum(r["is_refusal"] for r in cat_records) / len(cat_records), 3),
            "latency_end_to_end_ms": {"mean": cat_e2e["mean"], "p95": cat_e2e["p95"]} if cat_e2e else None,
        }

    print("\n" + "=" * 50)
    print("📊 BÁO CÁO TỔNG KẾT (corpus luật)")
    print("=" * 50)
    print(f"Tổng số câu hỏi: {summary['total_questions']}")
    print("\n-- Retrieval (chỉ tính câu hỏi QA, loại trừ summary) --")
    for k, v in summary["retrieval"].items():
        print(f"  {k:<28}: {v}")
    print("\n-- Generation (RAGAS-style, toàn bộ câu hỏi) --")
    for k, v in summary["generation"].items():
        print(f"  {k:<28}: {v}")
    print("\n-- Refusal rate (rule-based, không qua judge) --")
    print(f"  refusal_rate_overall        : {summary['refusal']['refusal_rate_overall']}")
    print(
        f"  refuse dù hit_rate=1        : {summary['refusal']['refusal_count_despite_hit']}"
        f"/{summary['refusal']['refusal_count_despite_hit_total_non_negative_qa']}"
    )
    print("\n-- Hiệu năng / Response time (ms, đã warm-up, KHÔNG tính judge) --")
    print(f"  {'stage':<18}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   n")
    for k in ("preprocessing_ms", "retrieval_ms", "rerank_ms", "generation_ms", "end_to_end_ms"):
        s = summary["latency"].get(k)
        if s:
            print(f"  {k:<18}{s['mean']:>9}{s['p50']:>9}{s['p95']:>9}{s['p99']:>9}{s['max']:>9}{s['n']:>4}")
    print("\n-- Theo category --")
    for cat, stats in summary["by_category"].items():
        print(
            f"  {cat:<12} (n={stats['count']:<3}) Faithfulness={stats['faithfulness']} "
            f"Relevancy={stats['answer_relevancy']} RefusalRate={stats['refusal_rate']}"
        )
    print("=" * 50)

    run_config = {
        "corpus": "legal_vn",
        "retrieval_strategy": strategy,
        "generator_model": local_llm_model,
        "embedding_model": os.getenv("LOCAL_EMBEDDING_MODEL") or "BAAI/bge-m3 (default)",
        "reranker_top_k": 3,
        "slm_enabled": slm_enabled,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "warm_up_seconds": warm_secs,
        "device_note": "CPU-only (không GPU) — latency phản ánh cấu hình phần cứng máy chạy benchmark",
        "latency_excludes": "Không tính thời gian Gemini judge; đã warm-up model trước khi đo",
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(REPORTS_DIR, f"report_legal_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"run_config": run_config, "summary": summary, "details": detailed_results},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n💾 Báo cáo chi tiết đã lưu: {report_path}")


if __name__ == "__main__":
    run()
