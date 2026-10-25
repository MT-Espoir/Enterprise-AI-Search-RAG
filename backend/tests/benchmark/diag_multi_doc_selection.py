"""
diag_multi_doc_selection.py — M1 của multi_doc_selection_plan.md: đo OFFLINE (chỉ đọc, không đổi code production)
các cách CHỌN top-3 sau rerank và cách GỘP ứng viên cho câu so sánh / tổng hợp nhiều tài liệu.

Vì sao: chẩn đoán 2026-10-25 cho thấy với 17 câu nhóm multi_doc (bộ vague70), Decomposition đưa đủ tài liệu bắt buộc
vào ứng viên ở 82% câu nhưng top-3 theo điểm rerank chỉ còn 29% — nút thắt ở bước chọn, không phải SLM.

Đo trên CÙNG ứng viên / cùng điểm cross-encoder (header bật):
  Nguồn câu con : không tách · qwen2.5:3b (production, lấy từ report slm) · qwen2.5:7b (Ollama local, cache)
  Gộp ứng viên  : cũ (sort điểm retriever toàn cục → cap 15) · H3 (xoay vòng theo thứ hạng từng câu con → cap 15)
  Chọn top-3    : S0 top-3 điểm (production) · V1 mỗi tài liệu 1 chunk · V2 ≥ 2 tài liệu · V3 = V2 + ngưỡng δ ·
                  V4 xoay vòng theo câu con (rerank ứng viên với từng câu con)
Kích hoạt (D1): query_type == "comparison" (SLM) HOẶC _COMPLEX_SIGNAL_PATTERN; câu không kích hoạt dùng S0.
Kèm các câu đơn tài liệu bị regex kích hoạt NHẦM (bộ vague70 + bộ 131) để đo tác hại của V1–V3.

Chạy (snapshot có header, ef_search khớp production — xem build_benchmark_db_full.py):
  python backend/tests/benchmark/diag_multi_doc_selection.py --db <snapshot có header> \\
      --slm-report report_full_all_v70_header_ef500_<ts>.json
  python backend/tests/benchmark/diag_multi_doc_selection.py --db <snapshot có header> --harm-check
    (M1b: áp V1 cho MỌI câu bộ 131 + vague70, so với S0 — quyết định phạm vi kích hoạt)
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import replace

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb
import requests

from app.core import context_header
from app.core.query.prompts import QUERY_DECOMPOSE_SYSTEM_INSTRUCTION
from app.core.query.query_processor import _COMPLEX_SIGNAL_PATTERN
from app.core.rag_pipeline import _MULTI_RETRIEVE_CAP, RAGPipeline
from app.core.ranking import Reranker
from app.core.retrieval import BM25Index, build_retriever
from app.ingestion.embedder.local_embedder import LocalEmbedder
from tests.benchmark.export_review_sheet import SETS
from tests.benchmark.run_benchmark_full import SnapshotOps

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BENCH_DIR, "reports")
SUBQUERY_CACHE = os.path.join(REPORTS_DIR, "diag_subqueries_vague70.json")
OUTPUT_PATH = os.path.join(REPORTS_DIR, "diag_multi_doc_selection.json")
K = 3
# Điểm bge-reranker-v2-m3 qua CrossEncoder đã qua Sigmoid → thang [0, 1]
DELTAS = (0.05, 0.1, 0.2, 0.3, 0.5)
LARGER_SLM = "qwen2.5:7b-instruct"


# ── Gộp ứng viên ─────────────────────────────────────────────────────────────

def merge_round_robin(per_sub: list[list], cap: int = _MULTI_RETRIEVE_CAP) -> list:
    """H3: hạng 1 của mọi câu con, rồi hạng 2... (bỏ trùng) tới cap — câu con nào cũng có đại diện."""
    merged, seen = [], set()
    for rank in range(max((len(x) for x in per_sub), default=0)):
        for hits in per_sub:
            if rank < len(hits) and hits[rank].chunk_id not in seen:
                merged.append(hits[rank])
                seen.add(hits[rank].chunk_id)
                if len(merged) == cap:
                    return merged
    return merged


# ── Chọn top-k (đầu vào: danh sách (chunk, điểm) đã sort giảm dần) ─────────────

def _norm_hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text).strip().lower().encode("utf-8")).hexdigest()


def dedupe_identical(ranked: list) -> list:
    """Bỏ chunk trùng NGUYÊN VĂN (bản trùng 45.signed ≡ Boluatlaodong2019) — giữ bản điểm cao nhất."""
    out, seen = [], set()
    for chunk, score in ranked:
        h = _norm_hash(chunk.text)
        if h not in seen:
            seen.add(h)
            out.append((chunk, score))
    return out


def _fill(picked: list, ranked: list, k: int) -> list:
    ids = {c.chunk_id for c in picked}
    for chunk, _ in ranked:
        if len(picked) == k:
            break
        if chunk.chunk_id not in ids:
            picked.append(chunk)
            ids.add(chunk.chunk_id)
    return picked


def select_s0(ranked: list, k: int = K) -> list:
    return [c for c, _ in ranked[:k]]


def select_v1(ranked: list, k: int = K) -> list:
    """Mỗi tài liệu 1 chunk (điểm cao nhất) trước, đủ k tài liệu rồi mới lấp."""
    ranked = dedupe_identical(ranked)
    picked, docs = [], set()
    for chunk, _ in ranked:
        if chunk.doc_id not in docs:
            picked.append(chunk)
            docs.add(chunk.doc_id)
        if len(picked) == k:
            return picked
    return _fill(picked, ranked, k)


def select_v2(ranked: list, k: int = K, delta: float | None = None) -> list:
    """Top-1 + chunk tốt nhất của tài liệu KHÁC (nếu delta: chỉ khi điểm ≥ top-1 − delta) + lấp theo điểm."""
    ranked = dedupe_identical(ranked)
    if not ranked:
        return []
    top_chunk, top_score = ranked[0]
    picked = [top_chunk]
    for chunk, score in ranked[1:]:
        if chunk.doc_id != top_chunk.doc_id:
            if delta is None or score >= top_score - delta:
                picked.append(chunk)
            break
    return _fill(picked, ranked, k)


def select_v4(per_sub_ranked: list[list], ranked: list, k: int = K) -> list:
    """Xoay vòng: chunk tốt nhất (chưa chọn) theo từng câu con; không có câu con → S0."""
    if len(per_sub_ranked) < 2:
        return select_s0(ranked, k)
    picked, ids, i = [], set(), 0
    queues = [[c for c, _ in dedupe_identical(r)] for r in per_sub_ranked]
    while len(picked) < k and any(queues):
        q = queues[i % len(queues)]
        while q and q[0].chunk_id in ids:
            q.pop(0)
        if q:
            chunk = q.pop(0)
            picked.append(chunk)
            ids.add(chunk.chunk_id)
        i += 1
    return _fill(picked, ranked, k)


# ── Chấm 1 lựa chọn ──────────────────────────────────────────────────────────

def score_selection(chunks: list, q: dict) -> dict:
    files = [c.filename for c in chunks]
    required = [set(g) for g in q.get("required_doc_groups", [])]
    relevant = [set(g) for g in q["relevant_doc_groups"]]
    return {
        "all_required": (float(all(any(f in g for f in files) for g in required)) if required else None),
        "doc_hit": float(any(f in g for g in relevant for f in files)),
        "chunk_hit": float(any(c.chunk_id in set(q["relevant_chunk_ids"]) for c in chunks)),
    }


def ollama_decompose(model: str, question: str) -> list[str]:
    r = requests.post(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')}/api/chat", timeout=600, json={
        "model": model, "stream": False, "format": "json", "options": {"temperature": 0.1},
        "messages": [{"role": "system", "content": QUERY_DECOMPOSE_SYSTEM_INSTRUCTION}, {"role": "user", "content": question}]})
    data = json.loads(r.json()["message"]["content"])
    return [str(s).strip() for s in (data.get("sub_queries") or []) if str(s).strip()]


def _setup(db_path: str):
    """Pipeline giống production (header bật): retriever hybrid top-10, reranker chấm TOÀN BỘ ứng viên."""
    context_header.set_enabled(True)
    collection = chromadb.PersistentClient(path=db_path).get_collection("benchmark_full")
    ef = ((collection.configuration or {}).get("hnsw") or {}).get("ef_search")
    ops = SnapshotOps(collection)
    bm25 = BM25Index()
    bm25.refresh(ops)
    retriever = build_retriever("hybrid", ops=ops, embedder=LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL")),
                               bm25_index=bm25, top_k=10)
    reranker = Reranker(top_k=10_000)
    pipeline = RAGPipeline(retriever=retriever, reranker=reranker, generator=None, query_processor=None,
                           observability_enabled=False, decomposition_enabled=True)

    def rank(question, pool):
        out = reranker.rerank(question, [replace(c, metadata=dict(c.metadata)) for c in pool])
        return [(c, c.metadata["rerank_score"]) for c in out]

    return retriever, reranker, pipeline, rank, ef


def run_harm_check(db_path: str):
    """M1b (quyết định D5): áp V1 cho TOÀN BỘ câu của bộ 131 + bộ vague70 (không tách câu) — V1 có làm hại câu chỉ
    cần 1 tài liệu không? So doc hit@3 / chunk hit@3 với S0 theo bộ, stratum, category; liệt kê câu đổi kết quả."""
    sets = {name: json.load(open(SETS[key]["output"], encoding="utf-8"))["questions"]
            for name, key in (("131 câu", "full"), ("vague70", "vague70"))}
    retriever, _, _, rank, ef = _setup(db_path)
    print(f"📋 M1b: {sum(len(v) for v in sets.values())} câu, ef_search={ef}\n", flush=True)
    rows = []
    for set_name, questions in sets.items():
        for i, q in enumerate(questions, 1):
            ranked = rank(q["question"], retriever.retrieve(q["question"], acl_bypass=True))
            s0, v1 = select_s0(ranked), select_v1(ranked)
            rows.append({"set": set_name, "id": q["id"], "stratum": q["stratum"], "category": q["category"],
                         "question": q["question"], "S0": score_selection(s0, q), "V1": score_selection(v1, q),
                         "S0_files": [c.filename for c in s0], "V1_files": [c.filename for c in v1],
                         "n_docs_top3_S0": len({c.doc_id for c in s0})})
            if i % 20 == 0:
                print(f"   {set_name} {i}/{len(questions)}", flush=True)

    def report(label, group):
        n = len(group)
        cells = []
        for key in ("doc_hit", "chunk_hit"):
            s0 = sum(r["S0"][key] for r in group) / n
            v1 = sum(r["V1"][key] for r in group) / n
            up = sum(r["V1"][key] > r["S0"][key] for r in group)
            down = sum(r["V1"][key] < r["S0"][key] for r in group)
            cells.append(f"{key} {s0:.3f}→{v1:.3f} (+{up}/−{down})")
        print(f"  {label:<34} n={n:<4} " + " · ".join(cells))

    print("\n=== V1 áp cho MỌI câu (không tách) — so với S0 ===")
    for set_name in sets:
        group = [r for r in rows if r["set"] == set_name]
        report(f"[{set_name}] TỔNG", group)
        for field in ("stratum", "category"):
            for value in sorted({r[field] for r in group}):
                report(f"   {field}={value}", [r for r in group if r[field] == value])
    single = [r for r in rows if r["category"] != "multi_doc"]
    report("TẤT CẢ câu KHÔNG phải multi_doc", single)
    print(f"\nTop-3 S0 chỉ gồm 1 tài liệu: {sum(r['n_docs_top3_S0'] == 1 for r in single)}/{len(single)} câu không phải multi_doc")
    print("\n=== Câu V1 làm TỆ ĐI (doc hit hoặc chunk hit) ===")
    for r in rows:
        if r["V1"]["doc_hit"] < r["S0"]["doc_hit"] or r["V1"]["chunk_hit"] < r["S0"]["chunk_hit"]:
            print(f"  [{r['set']}] {r['id']} {r['category']} doc {r['S0']['doc_hit']:.0f}→{r['V1']['doc_hit']:.0f} "
                  f"chunk {r['S0']['chunk_hit']:.0f}→{r['V1']['chunk_hit']:.0f} | {r['question'][:90]}")
            print(f"       S0 {r['S0_files']}\n       V1 {r['V1_files']}")
    out = os.path.join(REPORTS_DIR, "diag_multi_doc_harm_check.json")
    json.dump({"ef_search": ef, "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"💾 {out}")


def run(db_path: str, slm_report: str):
    multi = [q for q in json.load(open(SETS["vague70"]["output"], encoding="utf-8"))["questions"] if q["category"] == "multi_doc"]
    false_trig = [q for path in (SETS["vague70"]["output"], SETS["full"]["output"])
                  for q in json.load(open(path, encoding="utf-8"))["questions"]
                  if q.get("category") != "multi_doc" and _COMPLEX_SIGNAL_PATTERN.search(q["question"])]
    slm = json.load(open(os.path.join(REPORTS_DIR, slm_report), encoding="utf-8"))["modes"]["slm"]["details"]
    plan_3b = {d["id"]: d["query_plan"] for d in slm}
    missing = [q["id"] for q in multi if q["id"] not in plan_3b]
    if missing:
        sys.exit(f"❌ report slm thiếu câu {missing}")

    cache = json.load(open(SUBQUERY_CACHE, encoding="utf-8")) if os.path.exists(SUBQUERY_CACHE) else {}
    cache.setdefault(LARGER_SLM, {})
    for q in multi:
        if q["id"] not in cache[LARGER_SLM]:
            cache[LARGER_SLM][q["id"]] = ollama_decompose(LARGER_SLM, q["question"])
            json.dump(cache, open(SUBQUERY_CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    sources = {
        "không tách": {q["id"]: [] for q in multi},
        "qwen2.5:3b (production)": {q["id"]: plan_3b[q["id"]]["sub_queries"] for q in multi},
        "qwen2.5:7b": cache[LARGER_SLM],
    }
    triggered = {q["id"]: plan_3b[q["id"]]["query_type"] == "comparison" or bool(_COMPLEX_SIGNAL_PATTERN.search(q["question"]))
                 for q in multi}

    retriever, reranker, pipeline, rank, ef = _setup(db_path)
    print(f"📋 {len(multi)} câu multi_doc (kích hoạt D1: {sum(triggered.values())}) + {len(false_trig)} câu kích hoạt NHẦM · "
          f"ef_search={ef}\n", flush=True)

    variants = ["S0", "V1", "V2"] + [f"V3 δ={d}" for d in DELTAS] + ["V4"]
    results, gaps, t_select = {}, [], []
    for src_name, subs_by_id in sources.items():
        for merge in ("cũ", "H3"):
            if src_name == "không tách" and merge == "H3":
                continue
            rows = []
            for q in multi:
                subs = subs_by_id.get(q["id"], [])
                if len(subs) >= 2:
                    per_sub = [retriever.retrieve(s, acl_bypass=True) for s in subs]
                    pool = (merge_round_robin(per_sub) if merge == "H3" else pipeline._retrieve_multi(subs, acl_bypass=True))
                else:
                    pool = retriever.retrieve(q["question"], acl_bypass=True)
                ranked = rank(q["question"], pool)
                per_sub_ranked = [rank(s, pool) for s in subs] if len(subs) >= 2 else []
                t0 = time.perf_counter()
                picks = {"S0": select_s0(ranked), "V1": select_v1(ranked), "V2": select_v2(ranked)}
                for d in DELTAS:
                    picks[f"V3 δ={d}"] = select_v2(ranked, delta=d)
                picks["V4"] = select_v4(per_sub_ranked, ranked)
                t_select.append((time.perf_counter() - t0) * 1000 / len(picks))
                top_doc = ranked[0][0].doc_id
                other = next((s for c, s in ranked if c.doc_id != top_doc), None)
                if other is not None:
                    gaps.append(ranked[0][1] - other)
                pool_ok = score_selection([c for c, _ in ranked], q)["all_required"]
                row = {"id": q["id"], "pool_all_required": pool_ok, "triggered": triggered[q["id"]]}
                for v, chunks in picks.items():
                    row[v] = score_selection(chunks, q)
                    # Theo D1: câu không kích hoạt dùng S0
                    row[v + " (D1)"] = row[v] if triggered[q["id"]] else score_selection(picks["S0"], q)
                rows.append(row)
            results[f"{src_name} · gộp {merge}"] = rows
            print(f"✅ {src_name} · gộp {merge}", flush=True)

    # Câu đơn tài liệu bị regex kích hoạt nhầm: không tách (none/fast) — biến thể có làm mất đáp án không?
    ft_rows = []
    for q in false_trig:
        ranked = rank(q["question"], retriever.retrieve(q["question"], acl_bypass=True))
        row = {"id": q["id"], "question": q["question"]}
        for v, chunks in {"S0": select_s0(ranked), "V1": select_v1(ranked), "V2": select_v2(ranked),
                          **{f"V3 δ={d}": select_v2(ranked, delta=d) for d in DELTAS}}.items():
            row[v] = score_selection(chunks, q)
        ft_rows.append(row)

    n = len(multi)
    mean = lambda rows, v, key: sum(r[v][key] for r in rows) / len(rows)
    print(f"\n=== 17 câu multi_doc: tỷ lệ lấy ĐỦ mọi tài liệu bắt buộc trong top-3 (áp cho MỌI câu / theo D1) ===")
    print(f"{'cấu hình':<36}{'ứng viên':>9}" + "".join(f"{v:>12}" for v in variants))
    for name, rows in results.items():
        pool = sum(r["pool_all_required"] for r in rows) / n
        print(f"{name:<36}{pool:>9.3f}" + "".join(f"{mean(rows, v, 'all_required'):>6.3f}/{mean(rows, v + ' (D1)', 'all_required'):<5.3f}" for v in variants))
    print("\n=== doc hit@3 / chunk hit@3 (theo D1) — không được thấp hơn S0 ===")
    for name, rows in results.items():
        print(f"{name:<36}" + "".join(f"{mean(rows, v + ' (D1)', 'doc_hit'):>6.3f}/{mean(rows, v + ' (D1)', 'chunk_hit'):<5.3f}" for v in variants))
    print(f"\n=== {len(ft_rows)} câu đơn tài liệu bị regex kích hoạt NHẦM (không tách) — doc hit@3 / chunk hit@3 ===")
    for r in ft_rows:
        print(f"  {r['id']}: " + "  ".join(f"{v}={r[v]['doc_hit']:.0f}/{r[v]['chunk_hit']:.0f}" for v in variants if v in r)
              + f"  | {r['question'][:70]}")
    gaps.sort()
    print(f"\nKhoảng cách điểm top-1 → chunk tốt nhất tài liệu khác (n={len(gaps)}): "
          f"min {gaps[0]:.2f} · trung vị {gaps[len(gaps) // 2]:.2f} · max {gaps[-1]:.2f}")
    print(f"Thời gian chọn TB {sum(t_select) / len(t_select):.3f} ms/biến thể/câu")
    json.dump({"ef_search": ef, "variants": variants, "results": results, "false_triggered": ft_rows, "gaps": gaps},
              open(OUTPUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"💾 {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Snapshot có header (ef_search khớp production)")
    parser.add_argument("--slm-report", help="Report chế độ slm 17 câu multi_doc (câu con 3b + query_type)")
    parser.add_argument("--harm-check", action="store_true",
                        help="M1b: áp V1 cho toàn bộ bộ 131 + vague70 (không tách), so với S0")
    args = parser.parse_args()
    if args.harm_check:
        run_harm_check(args.db)
    elif args.slm_report:
        run(args.db, args.slm_report)
    else:
        parser.error("cần --slm-report (M1) hoặc --harm-check (M1b)")
