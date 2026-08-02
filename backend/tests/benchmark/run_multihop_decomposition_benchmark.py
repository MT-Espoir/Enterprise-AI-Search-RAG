"""
run_multihop_decomposition_benchmark.py — Phase 4 mục 2: đo tác động thật của
Decomposition/Multi-Query (query_processor.decompose() + RAGPipeline._retrieve_multi())
trên ĐÚNG 3 câu hỏi category="multi_hop" trong evaluation_dataset_legal.json —
đây là category yếu nhất ở Phase 3(a) (faithfulness=0.667, thấp nhất, xem
rag_core_quality_roadmap.md mục 6e).

So sánh 2 điều kiện trên CÙNG 3 câu hỏi, dùng benchmark_chromadb_legal đã build
sẵn (không build DB riêng, không đụng evaluation_dataset_legal.json/report cũ):
  - baseline: ép query_type="simple" (không decompose) — mô phỏng hành vi TRƯỚC
    Phase 4 mục 2, tương đương flow QA thường trong run_full_benchmark_legal.py.
  - decomposition: dùng QueryProcessor.process() thật (classify) + decompose()
    thật khi query_type in (complex/comparison/reasoning) — hành vi SAU khi sửa.

Chạy: python backend/tests/benchmark/run_multihop_decomposition_benchmark.py
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
from app.core.bm25_index import BM25Index
from app.core.retriever_factory import build_retriever
from app.core.query_processor import QueryProcessor
from app.core.reranker import Reranker
from app.core.local_generator import LocalGenerator
from app.core.rag_pipeline import RAGPipeline

from tests.benchmark.metrics import calculate_hit_rate, calculate_mrr, calculate_recall_at_k
from tests.benchmark.ragas_metrics import score_faithfulness, score_answer_relevancy, GEMINI_JUDGE_MODEL_DEFAULT

DB_PATH = os.path.join(os.path.dirname(__file__), "benchmark_chromadb_legal")
COLLECTION_NAME = "benchmark_legal"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
LOCAL_GENERATOR_MODEL_DEFAULT = "qwen2.5:3b-instruct"

CONDITIONS = ["baseline_no_decompose", "decomposition"]


class MockVectorStoreOps(VectorStoreOps):
    def __init__(self, collection):
        self.collection = collection


def _mean(values):
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 3) if clean else None


def run():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Thiếu GOOGLE_API_KEY trong .env")
        sys.exit(1)

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL_DEFAULT)
    local_llm_model = os.getenv("LOCAL_LLM_MODEL", LOCAL_GENERATOR_MODEL_DEFAULT)
    judge_model = os.getenv("GEMINI_JUDGE_MODEL", GEMINI_JUDGE_MODEL_DEFAULT)

    try:
        requests.get(f"{ollama_base_url.rstrip('/')}/api/tags", timeout=5).raise_for_status()
    except Exception:
        print(f"❌ Không kết nối được Ollama tại {ollama_base_url}.")
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    vector_ops = MockVectorStoreOps(collection)
    embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL"))

    bm25_index = BM25Index()
    bm25_index.refresh(vector_ops)
    retriever = build_retriever("hybrid", ops=vector_ops, embedder=embedder, bm25_index=bm25_index, top_k=10)

    query_processor = QueryProcessor(base_url=ollama_base_url, model_name=local_llm_model,
                                      default_strategy="hybrid", default_top_k=10, slm_enabled=True)
    reranker = Reranker.get_instance(top_k=3)
    generator = LocalGenerator(base_url=ollama_base_url, model_name=local_llm_model)
    rag = RAGPipeline(retriever=retriever, reranker=reranker, generator=generator,
                       query_processor=query_processor, observability_enabled=False)

    dataset_path = os.path.join(os.path.dirname(__file__), "evaluation_dataset_legal.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    multi_hop_items = [item for item in dataset if item.get("category") == "multi_hop"]
    print(f"🚀 Phase 4 mục 2 — Benchmark Decomposition trên {len(multi_hop_items)} câu multi_hop "
          f"× {len(CONDITIONS)} điều kiện...\n")

    detailed_results = []

    for idx, item in enumerate(multi_hop_items):
        question = item["question"]
        expected = item["expected_sources"]
        print(f"\n[{idx + 1}/{len(multi_hop_items)}] Q: {question}")

        for condition in CONDITIONS:
            if condition == "baseline_no_decompose":
                candidates = retriever.retrieve(question, vector_query=None)
                sub_queries_used = []
            else:
                plan = query_processor.process(question)
                sub_queries_used = []
                if plan.query_type in ("complex", "comparison", "reasoning"):
                    sub_queries_used = query_processor.decompose(
                        plan.rewritten_query if plan.rewrite else question
                    )
                if len(sub_queries_used) >= 2:
                    candidates = rag._retrieve_multi(sub_queries_used, doc_id=None)
                else:
                    candidates = retriever.retrieve(question, vector_query=None)

            if not candidates:
                answer, top_chunks = "Tôi không tìm thấy tài liệu nào liên quan đến câu hỏi của bạn.", []
            else:
                top_chunks = reranker.rerank(question, candidates)
                gen_result = generator.generate(question, top_chunks)
                answer = gen_result.answer

            h_rate = calculate_hit_rate(candidates, expected)
            mrr = calculate_mrr(candidates, expected)
            r3 = calculate_recall_at_k(candidates, expected, 3)
            r5 = calculate_recall_at_k(candidates, expected, min(5, len(candidates) or 1))
            hit_post_rerank = calculate_hit_rate(top_chunks, expected)

            top_chunk_texts = [c.text for c in top_chunks] if candidates else []
            faithfulness = score_faithfulness(question, answer, top_chunk_texts, api_key=api_key, model_name=judge_model)
            relevancy = score_answer_relevancy(question, answer, api_key=api_key, is_negative_expected=False, model_name=judge_model)

            record = {
                "question": question,
                "condition": condition,
                "sub_queries": sub_queries_used,
                "n_candidates": len(candidates),
                "answer": answer,
                "retrieval_metrics": {"hit_rate": h_rate, "mrr": mrr, "recall@3": r3, "recall@5": r5,
                                       "hit_rate_post_rerank@3": hit_post_rerank},
                "ragas": {"faithfulness": faithfulness.get("score"),
                          "faithfulness_reasoning": faithfulness.get("reasoning"),
                          "answer_relevancy": relevancy.get("score")},
            }
            detailed_results.append(record)

            print(f"  [{condition:^22}] n_candidates={len(candidates)} sub_queries={sub_queries_used}")
            print(f"    Hit={h_rate} MRR={mrr:.2f} Faithfulness={record['ragas']['faithfulness']} "
                  f"Relevancy={record['ragas']['answer_relevancy']}")
            print(f"    A: {answer[:200]}{'...' if len(answer) > 200 else ''}")

    print("\n" + "=" * 60)
    print("📊 BÁO CÁO TỔNG KẾT — Phase 4 mục 2 Decomposition (multi_hop)")
    print("=" * 60)
    for condition in CONDITIONS:
        cond_records = [r for r in detailed_results if r["condition"] == condition]
        print(f"\n-- {condition} --")
        print(f"  hit_rate       : {_mean([r['retrieval_metrics']['hit_rate'] for r in cond_records])}")
        print(f"  mrr            : {_mean([r['retrieval_metrics']['mrr'] for r in cond_records])}")
        print(f"  recall@3       : {_mean([r['retrieval_metrics']['recall@3'] for r in cond_records])}")
        print(f"  faithfulness   : {_mean([r['ragas']['faithfulness'] for r in cond_records])}")
        print(f"  answer_relevancy: {_mean([r['ragas']['answer_relevancy'] for r in cond_records])}")
    print("=" * 60)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(REPORTS_DIR, f"report_multihop_decompose_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"details": detailed_results}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Báo cáo chi tiết đã lưu: {report_path}")


if __name__ == "__main__":
    run()
