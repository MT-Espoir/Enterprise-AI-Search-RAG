"""
generate_rag_answers.py — Bước 1 của đánh giá câu trả lời bằng `ragas` chính thức (run_ragas_eval.py là bước 2).

Chạy ĐÚNG RAGPipeline.run() của production (QueryProcessor SLM + Decomposition + Rerank top-3 + guardrails +
Generator theo LLM_PROVIDER trong .env) trên snapshot benchmark, ghi lại câu trả lời và đúng các đoạn
Generator đã thấy (sau Retrieval Guardrail, nhãn nguồn có header ngữ cảnh như _format_context).

Tách 2 bước vì `ragas` xung đột phiên bản với ml_env (aiohttp/jiter/docstring-parser) → chấm điểm chạy trong
venv riêng, bước này chạy trong ml_env. Ghi JSONL từng câu, chạy lại thì bỏ qua câu đã có (Generator lỗi
quota → câu đó không được ghi, lần sau chạy bù).

Chạy (ml_env):
  python backend/tests/benchmark/generate_rag_answers.py --db <snapshot ef_search=500> \\
      --dataset evaluation_dataset_full.json --dataset evaluation_dataset_vague70.json --out reports/rag_answers.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb

from app.core import context_header
from app.core.generation import build_generator
from app.core.query import QueryProcessor
from app.core.rag_pipeline import RAGPipeline
from app.core.ranking import Reranker
from app.core.retrieval import BM25Index, build_retriever
from app.ingestion.embedder.local_embedder import LocalEmbedder
from tests.benchmark.run_benchmark_full import BENCH_DIR, COLLECTION_NAME, MANIFEST_PATH, SnapshotOps

RETRIEVER_TOP_K = 10
RERANK_TOP_K = 3


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() == "true"


class RecordingGenerator:
    """Bọc Generator thật, ghi lại các đoạn được đưa vào generate() (đúng context Generator thấy)."""

    def __init__(self, inner):
        self.inner = inner
        self.last_chunks = []

    def generate(self, question, chunks, history=None):
        self.last_chunks = list(chunks)
        return self.inner.generate(question, chunks, history=history)


def context_strings(chunks) -> list[str]:
    """Giống nhãn nguồn của BaseGenerator._format_context (header ngữ cảnh ở nhãn, nội dung giữ nguyên)."""
    out = []
    for c in chunks:
        page = f" (Trang {c.page})" if c.page else ""
        header = context_header.header_of(c.metadata)
        out.append(f"Nguồn: {c.filename}{page}{f' | {header}' if header else ''}\n{c.text}")
    return out


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) or os.path.exists(path) else os.path.join(BENCH_DIR, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Snapshot ChromaDB (bản sao ef_search=500)")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--ids", help="Chỉ chạy các id này (phân cách dấu phẩy)")
    parser.add_argument("--limit", type=int, help="Chỉ chạy N câu đầu (chạy thử)")
    parser.add_argument("--out", required=True, help="File JSONL kết quả (chạy lại sẽ bỏ qua câu đã có)")
    args = parser.parse_args()

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    collection = chromadb.PersistentClient(path=args.db).get_collection(args.collection)
    db_fingerprint = hashlib.sha256("\n".join(sorted(collection.get(include=[])["ids"])).encode("utf-8")).hexdigest()
    if db_fingerprint != manifest["fingerprint"]:
        sys.exit(f"❌ Chunk trong {args.db} không khớp fingerprint snapshot.")

    questions = []
    for path in args.dataset:
        with open(_resolve(path), encoding="utf-8") as f:
            dataset = json.load(f)
        if dataset["snapshot_fingerprint"] != manifest["fingerprint"]:
            sys.exit(f"❌ {path} tạo trên snapshot khác.")
        name = os.path.splitext(os.path.basename(path))[0].replace("evaluation_dataset_", "")
        questions += [{**q, "dataset": name} for q in dataset["questions"]
                      if args.split == "all" or q["split"] == args.split]
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = questions[:args.limit]

    out_path = _resolve(args.out)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [q for q in questions if q["id"] not in done]
    print(f"📋 {len(questions)} câu, đã có {len(questions) - len(todo)}, cần chạy {len(todo)}")

    # Cấu hình như chat.get_rag_pipeline() (đọc cùng biến môi trường .env)
    header_enabled = _env_flag("CONTEXT_HEADER_ENABLED", False)
    context_header.set_enabled(header_enabled)  # trước BM25 refresh
    ops = SnapshotOps(collection)
    embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL"))
    strategy = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
    bm25_index = None
    if strategy == "hybrid":
        bm25_index = BM25Index()
        bm25_index.refresh(ops)
    retriever = build_retriever(strategy, ops=ops, embedder=embedder, bm25_index=bm25_index, top_k=RETRIEVER_TOP_K)
    config = {k: os.getenv(k) for k in ("LLM_PROVIDER", "GOOGLE_API_KEY", "GEMINI_MODEL", "ANTHROPIC_API_KEY",
                                         "ANTHROPIC_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
                                         "OLLAMA_BASE_URL", "LOCAL_LLM_MODEL")}
    generator = RecordingGenerator(build_generator(config))
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=Reranker(top_k=RERANK_TOP_K),
        generator=generator,
        query_processor=QueryProcessor(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model_name=os.getenv("LOCAL_LLM_MODEL"),
            default_strategy=strategy, default_top_k=RETRIEVER_TOP_K,
            slm_enabled=_env_flag("QUERY_PROCESSOR_SLM_ENABLED", True),
            decompose_by_entity=_env_flag("DECOMPOSE_BY_ENTITY_ENABLED", False),
        ),
        observability_enabled=False,
        decomposition_enabled=_env_flag("DECOMPOSITION_ENABLED", True),
        retrieval_guardrails_enabled=_env_flag("RETRIEVAL_GUARDRAILS_ENABLED", True),
        output_guardrails_enabled=_env_flag("OUTPUT_GUARDRAILS_ENABLED", True),
        diversity_enabled=_env_flag("RERANK_DIVERSITY_ENABLED", False),
    )
    run_config = {
        "llm_provider": config["LLM_PROVIDER"],
        "generator_model": config.get("GEMINI_MODEL") if config["LLM_PROVIDER"] == "gemini" else None,
        "slm_model": os.getenv("LOCAL_LLM_MODEL"),
        "context_header_enabled": header_enabled,
        "diversity_enabled": pipeline.diversity_enabled,
        "hnsw_ef_search": ((collection.configuration or {}).get("hnsw") or {}).get("ef_search"),
        "snapshot_fingerprint": manifest["fingerprint"],
    }
    print(f"⚙️  {json.dumps(run_config, ensure_ascii=False)}")

    failed = []
    with open(out_path, "a", encoding="utf-8") as out:
        for idx, q in enumerate(todo, 1):
            generator.last_chunks = []
            t0 = time.perf_counter()
            try:
                result = pipeline.run(q["question"], acl_bypass=True)
            except Exception as exc:  # Generator hết quota/lỗi mạng → bỏ qua, lần sau chạy bù
                failed.append(q["id"])
                print(f"[{idx}/{len(todo)}] {q['id']} ❌ {type(exc).__name__}: {str(exc)[:120]}")
                continue
            record = {
                "id": q["id"], "dataset": q["dataset"], "split": q["split"], "category": q["category"],
                "stratum": q["stratum"], "question": q["question"], "reference_answer": q.get("reference_answer"),
                "answer": result["answer"], "contexts": context_strings(generator.last_chunks),
                "context_chunk_ids": [c.chunk_id for c in generator.last_chunks],
                "finish_reason": result["finish_reason"], "guardrail": result["guardrail"],
                "seconds": round(time.perf_counter() - t0, 1), "run_config": run_config,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{idx}/{len(todo)}] {q['id']} {result['finish_reason']} · {len(record['contexts'])} đoạn · "
                  f"{record['seconds']}s")
    print(f"\n💾 {out_path} — lỗi {len(failed)}: {failed}")


if __name__ == "__main__":
    main()
