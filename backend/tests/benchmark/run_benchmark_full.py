"""
run_benchmark_full.py — Benchmark RETRIEVAL tổng quát trên snapshot 89 tài liệu
(benchmark_chromadb_full/, evaluation_dataset_full.json): Hit Rate, Recall@k, MRR,
Precision@k ở mức chunk VÀ mức tài liệu, trước/sau rerank, kèm latency từng stage.

KHÔNG chạy Generator — chỉ đo phần tìm kiếm, nên nhanh hơn nhiều so với
run_full_benchmark_legal.py và không cần judge LLM.

Đi đúng đường retrieval của production (RAGPipeline.run, nhánh QA):
QueryProcessor → (Rewrite | Decomposition) → Retriever → Reranker. Chạy theo từng
CHẾ ĐỘ QueryProcessor để tách bạch đóng góp của từng tầng:
  - none : bỏ qua QueryProcessor, đưa thẳng câu hỏi vào retriever (baseline retrieval thuần)
  - fast : QueryProcessor chỉ fast-path (SLM tắt)
  - slm  : QueryProcessor đầy đủ (SLM classify/rewrite + Decomposition) — cần Ollama

ACL: luôn acl_bypass=True (góc nhìn admin) — benchmark đo chất lượng tìm kiếm,
không đo phân quyền.

Chạy:
  python backend/tests/benchmark/run_benchmark_full.py                 # split=dev, mọi chế độ
  python backend/tests/benchmark/run_benchmark_full.py --split test --modes none,slm
  python backend/tests/benchmark/run_benchmark_full.py --db .../benchmark_chromadb_full_ctx_all --context-header
    (--context-header: bật killswitch CONTEXT_HEADER_ENABLED để BM25/Reranker thấy metadata
     context_header — dùng cho snapshot lưu header kiểu production, poc_contextual_headers_all.py)
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb
import requests

from app.ingestion.embedder.local_embedder import LocalEmbedder
from app.vectorstore.operations import VectorStoreOps
from app.core.retrieval import BM25Index, build_retriever
from app.core.query import QueryProcessor
from app.core.ranking import Reranker
from app.core.rag_pipeline import RAGPipeline
from app.core import context_header

from tests.benchmark.metrics import (
    bootstrap_ci,
    chunk_level_metrics,
    doc_level_metrics,
    reciprocal_rank,
)
from tests.benchmark.latency import StageTimer, summarize_latency

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BENCH_DIR, "benchmark_chromadb_full")
COLLECTION_NAME = "benchmark_full"
MANIFEST_PATH = os.path.join(BENCH_DIR, "snapshot_manifest_full.json")
DATASET_PATH = os.path.join(BENCH_DIR, "evaluation_dataset_full.json")
REPORTS_DIR = os.path.join(BENCH_DIR, "reports")

OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
LOCAL_LLM_MODEL_DEFAULT = "qwen2.5:3b-instruct"
RETRIEVER_TOP_K = 10
K_VALUES = (1, 3, 5, 10)
# Rerank TOÀN BỘ candidates (không cắt top 3 như production) để đo được @1/@3/@5 sau
# rerank; @3 sau rerank chính là thứ Generator production nhận được.
RERANK_K_VALUES = (1, 3, 5)
PRODUCTION_RERANK_TOP_K = 3
MODES = ("none", "fast", "slm")
WARM_QUESTION = "Điều kiện thành lập doanh nghiệp là gì?"


class SnapshotOps(VectorStoreOps):
    def __init__(self, collection):
        self.collection = collection


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 3) if clean else None


def score_ranking(chunk_ids: list[str], filenames: list[str], relevant_ids: set[str],
                  relevant_groups: list[set[str]], k_values: tuple[int, ...]) -> dict:
    relevant_files = set().union(*relevant_groups)
    chunk = {"mrr": reciprocal_rank(chunk_ids, lambda cid: cid in relevant_ids)}
    doc = {"mrr": reciprocal_rank(filenames, lambda f: f in relevant_files)}
    for k in k_values:
        chunk.update(chunk_level_metrics(chunk_ids, relevant_ids, k))
        doc.update(doc_level_metrics(filenames, relevant_groups, k))
    return {"chunk": chunk, "doc": doc}


def retrieve_like_production(mode: str, question: str, query_processor, pipeline: RAGPipeline,
                             retriever, reranker, timer: StageTimer):
    """Tái hiện RAGPipeline.run() nhánh QA tới hết Rerank (app/core/rag_pipeline.py)."""
    info = {"query_type": None, "rewrite": False, "rewritten_query": None,
            "sub_queries": [], "bypass_retrieval": False}
    vector_query = None
    sub_queries: list[str] = []

    if mode != "none":
        with timer.stage("preprocessing_ms"):
            plan = query_processor.process(question)
        info.update(query_type=plan.query_type, rewrite=plan.rewrite,
                    rewritten_query=plan.rewritten_query if plan.rewrite else None,
                    bypass_retrieval=plan.bypass_retrieval)
        if plan.bypass_retrieval:
            # Production không có doc_id → trả "Vui lòng chỉ định tài liệu", không retrieve
            return [], [], info
        effective_question = plan.rewritten_query if plan.rewrite else question
        if pipeline.decomposition_enabled and plan.query_type in ("complex", "comparison", "reasoning"):
            with timer.stage("preprocessing_ms"):
                sub_queries = query_processor.decompose(effective_question)
        vector_query = plan.rewritten_query if plan.rewrite else None
        info["sub_queries"] = sub_queries

    with timer.stage("retrieval_ms"):
        if len(sub_queries) >= 2:
            candidates = pipeline._retrieve_multi(sub_queries, acl_bypass=True)
        else:
            candidates = retriever.retrieve(question, vector_query=vector_query, acl_bypass=True)

    if not candidates:
        return [], [], info
    with timer.stage("rerank_ms"):
        reranked = reranker.rerank(question, candidates)
    return candidates, reranked, info


def summarize(records: list[dict]) -> dict:
    def block(level: str, stage: str) -> dict:
        keys = records[0]["metrics"][stage][level].keys()
        out = {}
        for key in keys:
            values = [r["metrics"][stage][level][key] for r in records]
            out[key] = {"mean": _mean(values), "ci95": bootstrap_ci(values)}
        return out

    summary = {
        "n": len(records),
        "retrieval": {"chunk": block("chunk", "retrieval"), "doc": block("doc", "retrieval")},
        "reranked": {"chunk": block("chunk", "reranked"), "doc": block("doc", "reranked")},
        "latency": summarize_latency([r["latency_ms"] for r in records]),
        "query_processor": {
            "rewrite_count": sum(r["query_plan"]["rewrite"] for r in records),
            "decomposed_count": sum(len(r["query_plan"]["sub_queries"]) >= 2 for r in records),
            "bypass_retrieval_count": sum(r["query_plan"]["bypass_retrieval"] for r in records),
        },
        "by_stratum": {},
        "by_category": {},
    }
    for field, target in (("stratum", "by_stratum"), ("category", "by_category")):
        groups = defaultdict(list)
        for r in records:
            groups[r[field]].append(r)
        for name, group in sorted(groups.items()):
            summary[target][name] = {
                "n": len(group),
                "doc_hit@3_reranked": _mean([r["metrics"]["reranked"]["doc"]["hit@3"] for r in group]),
                "chunk_hit@3_reranked": _mean([r["metrics"]["reranked"]["chunk"]["hit@3"] for r in group]),
                "doc_mrr_retrieval": _mean([r["metrics"]["retrieval"]["doc"]["mrr"] for r in group]),
                "chunk_recall@10_retrieval": _mean([r["metrics"]["retrieval"]["chunk"]["recall@10"] for r in group]),
            }
    return summary


def print_summary(mode: str, s: dict):
    def fmt(m):
        lo_hi = f" [{m['ci95'][0]:.3f}–{m['ci95'][1]:.3f}]" if m["ci95"] else ""
        return f"{m['mean']:.3f}{lo_hi}"

    print(f"\n{'=' * 64}\n📊 Chế độ QueryProcessor = {mode}  (n={s['n']}, CI 95% bootstrap)\n{'=' * 64}")
    for stage, label in (("retrieval", f"Retriever top-{RETRIEVER_TOP_K}"), ("reranked", "Sau Reranker")):
        for level in ("doc", "chunk"):
            metrics = s[stage][level]
            print(f"\n-- {label} · mức {level} --")
            for key, m in metrics.items():
                print(f"  {key:<14} {fmt(m)}")
    print("\n-- Latency (ms, đã warm-up, không có generation) --")
    print(f"  {'stage':<18}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   n")
    for key, st in s["latency"].items():
        if st:
            print(f"  {key:<18}{st['mean']:>9}{st['p50']:>9}{st['p95']:>9}{st['p99']:>9}{st['max']:>9}{st['n']:>4}")
    qp = s["query_processor"]
    print(f"\n-- QueryProcessor: rewrite={qp['rewrite_count']}, decomposition={qp['decomposed_count']}, "
          f"bypass={qp['bypass_retrieval_count']} --")
    print("\n-- Theo nhóm tài liệu (doc hit@3 sau rerank / chunk hit@3 sau rerank) --")
    for name, st in s["by_stratum"].items():
        print(f"  {name:<18} n={st['n']:<4} {st['doc_hit@3_reranked']}  /  {st['chunk_hit@3_reranked']}")
    print("\n-- Theo loại câu hỏi --")
    for name, st in s["by_category"].items():
        print(f"  {name:<18} n={st['n']:<4} {st['doc_hit@3_reranked']}  /  {st['chunk_hit@3_reranked']}")


def describe_context_headers(collection) -> dict:
    """Bộ header có trong snapshot — ghi vào run_config (fingerprint chỉ băm chunk_id, không
    phân biệt được embedding có header hay chưa). Chặn snapshot kiểu PoC cũ (header đã nằm
    trong document text) vì bật killswitch sẽ ghép header 2 lần."""
    data = collection.get(include=["documents", "metadatas"])
    header_of, doubled = {}, 0
    for text, meta in zip(data["documents"], data["metadatas"]):
        header = (meta.get("context_header") or "").strip()
        if header:
            header_of[meta["doc_id"]] = header
            doubled += text.startswith(header)
    if doubled:
        sys.exit(f"❌ {doubled} chunk đã có header trong document text (snapshot kiểu PoC cũ) — "
                 "bỏ --context-header, nếu không header bị ghép 2 lần.")
    payload = "\n".join(f"{doc_id}\x1f{header_of[doc_id]}" for doc_id in sorted(header_of))
    return {
        "chunks_with_header": sum(bool(m.get("context_header")) for m in data["metadatas"]),
        "documents_with_header": len(header_of),
        "headers_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def run(split: str, modes: list[str], dataset_path: str = DATASET_PATH, db_path: str = DB_PATH,
        use_context_header: bool = False):
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    if dataset["snapshot_fingerprint"] != manifest["fingerprint"]:
        sys.exit("❌ Dataset được tạo trên snapshot khác — nhãn chunk_id không còn đáng tin.")

    questions = [q for q in dataset["questions"] if split == "all" or q["split"] == split]
    print(f"📋 {len(questions)} câu hỏi (split={split}), chế độ: {', '.join(modes)}\n")

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL_DEFAULT)
    local_llm_model = os.getenv("LOCAL_LLM_MODEL", LOCAL_LLM_MODEL_DEFAULT)
    if "slm" in modes:
        try:
            requests.get(f"{ollama_base_url.rstrip('/')}/api/tags", timeout=5).raise_for_status()
        except Exception:
            sys.exit(f"❌ Chế độ slm cần Ollama tại {ollama_base_url} — chạy `ollama serve` hoặc bỏ slm khỏi --modes.")

    collection = chromadb.PersistentClient(path=db_path).get_collection(COLLECTION_NAME)
    ops = SnapshotOps(collection)
    header_info = None
    if use_context_header:
        header_info = describe_context_headers(collection)
        context_header.set_enabled(True)  # trước BM25 refresh: index tokenize "header + nội dung"
        print(f"🏷️  Header ngữ cảnh BẬT: {header_info['documents_with_header']} tài liệu / "
              f"{header_info['chunks_with_header']} chunk có header\n")
    embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL"))
    strategy = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
    bm25_index = None
    if strategy == "hybrid":
        bm25_index = BM25Index()
        bm25_index.refresh(ops)
    retriever = build_retriever(strategy, ops=ops, embedder=embedder, bm25_index=bm25_index, top_k=RETRIEVER_TOP_K)
    reranker = Reranker(top_k=10_000)
    decomposition_enabled = os.getenv("DECOMPOSITION_ENABLED", "true").lower() == "true"

    report = {
        "run_config": {
            "split": split,
            "modes": modes,
            "snapshot_fingerprint": manifest["fingerprint"],
            "db_path": os.path.relpath(db_path, BENCH_DIR),
            "corpus": f"{manifest['total_documents']} tài liệu / {manifest['total_chunks']} chunk",
            "retrieval_strategy": strategy,
            "retriever_top_k": RETRIEVER_TOP_K,
            "embedding_model": os.getenv("LOCAL_EMBEDDING_MODEL") or "BAAI/bge-m3 (default)",
            "slm_model": local_llm_model,
            "decomposition_enabled": decomposition_enabled,
            "production_rerank_top_k": PRODUCTION_RERANK_TOP_K,
            "acl_bypass": True,
            "context_header_enabled": use_context_header,
            "context_headers": header_info,
            "device_note": "CPU-only — latency phản ánh phần cứng máy chạy benchmark",
        },
        "modes": {},
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "" if os.path.abspath(db_path) == os.path.abspath(DB_PATH) else f"_{os.path.basename(os.path.normpath(db_path))}"
    report_path = os.path.join(REPORTS_DIR, f"report_full_{split}{tag}_{timestamp}.json")

    for mode in modes:
        query_processor = None
        if mode != "none":
            query_processor = QueryProcessor(
                base_url=ollama_base_url, model_name=local_llm_model,
                default_strategy=strategy, default_top_k=RETRIEVER_TOP_K, slm_enabled=(mode == "slm"),
            )
        pipeline = RAGPipeline(
            retriever=retriever, reranker=reranker, generator=None,
            query_processor=query_processor, observability_enabled=False,
            decomposition_enabled=decomposition_enabled,
        )

        print(f"🔥 [{mode}] Warm-up (không tính vào số đo)...")
        retrieve_like_production(mode, WARM_QUESTION, query_processor, pipeline, retriever, reranker, StageTimer())

        records = []
        for idx, q in enumerate(questions, 1):
            timer = StageTimer()
            candidates, reranked, plan_info = retrieve_like_production(
                mode, q["question"], query_processor, pipeline, retriever, reranker, timer,
            )
            relevant_ids = set(q["relevant_chunk_ids"])
            relevant_groups = [set(g) for g in q["relevant_doc_groups"]]
            metrics = {
                "retrieval": score_ranking([c.chunk_id for c in candidates], [c.filename for c in candidates],
                                           relevant_ids, relevant_groups, K_VALUES),
                "reranked": score_ranking([c.chunk_id for c in reranked], [c.filename for c in reranked],
                                          relevant_ids, relevant_groups, RERANK_K_VALUES),
            }
            records.append({
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "stratum": q["stratum"],
                "query_plan": plan_info,
                "retrieved": [{"chunk_id": c.chunk_id, "filename": c.filename, "page": c.page} for c in candidates],
                "reranked_top": [c.chunk_id for c in reranked[:PRODUCTION_RERANK_TOP_K]],
                "metrics": metrics,
                "latency_ms": timer.as_record(),
            })
            m = metrics["reranked"]
            print(f"[{mode} {idx}/{len(questions)}] {q['id']} doc@3={m['doc']['hit@3']:.0f} "
                  f"chunk@3={m['chunk']['hit@3']:.0f} | {timer.as_record()['end_to_end_ms']}ms")

        summary = summarize(records)
        report["modes"][mode] = {"summary": summary, "details": records}
        print_summary(mode, summary)
        # Ghi lại SAU MỖI chế độ: chế độ slm chạy ~1h trên CPU — bị dừng giữa chừng thì vẫn
        # giữ được kết quả các chế độ đã xong (đã mất trắng 1 lần khi chỉ ghi ở cuối)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã lưu ({', '.join(report['modes'])}): {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--modes", default=",".join(MODES),
                        help="Danh sách chế độ QueryProcessor, phân cách bằng dấu phẩy: none,fast,slm")
    parser.add_argument("--dataset", default=DATASET_PATH, help="Đường dẫn dataset (mặc định evaluation_dataset_full.json)")
    parser.add_argument("--db", default=DB_PATH,
                        help="ChromaDB snapshot (mặc định benchmark_chromadb_full; PoC: benchmark_chromadb_full_ctx)")
    parser.add_argument("--context-header", action="store_true",
                        help="Bật killswitch header ngữ cảnh (BM25/Reranker thấy metadata context_header)")
    args = parser.parse_args()
    selected = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(selected) - set(MODES)
    if unknown:
        parser.error(f"Chế độ không hợp lệ: {unknown}")
    run(split=args.split, modes=selected, dataset_path=args.dataset, db_path=args.db,
        use_context_header=args.context_header)
