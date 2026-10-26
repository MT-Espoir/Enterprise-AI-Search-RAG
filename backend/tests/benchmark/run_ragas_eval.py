"""
run_ragas_eval.py — Bước 2: chấm câu trả lời bằng thư viện `ragas` CHÍNH THỨC (thay judge tự viết ragas_metrics.py).

Đọc JSONL của generate_rag_answers.py, chấm từng câu bằng các metric `ragas.metrics.collections`:
  - faithfulness        : tách câu trả lời thành mệnh đề → NLI từng mệnh đề với context → tỷ lệ được hỗ trợ
  - answer_relevancy    : sinh ngược `strictness` câu hỏi từ câu trả lời → cosine embedding với câu hỏi gốc
  - context_recall      : mệnh đề của đáp án tham chiếu có nằm trong context không (đo truy xuất)
  - factual_correctness : mệnh đề câu trả lời vs đáp án tham chiếu (F1)
Judge = Qwen local qua API OpenAI-compatible của Ollama (mặc định biến thể num_ctx 8192 — mặc định 4096 của
Ollama cắt cụt prompt NLI có 3 đoạn tài liệu mà không báo lỗi). Embedding = BGE-M3 (cùng model production).

Tiền xử lý (ghi trong report, không đổi nội dung được chấm):
  - BỎ cụm trích nguồn "(Nguồn: x.pdf, Trang 3)" / "[Tài liệu 1, tr. 2]" khỏi câu trả lời trước khi chấm: chạy
    thử, Qwen 7b tách dòng trích nguồn thành 1 mệnh đề rồi chấm 0 → faithfulness bị trừ oan có hệ thống.
  - Câu TỪ CHỐI HẲN (MỞ ĐẦU bằng "Tôi không tìm thấy thông tin…") không chấm faithfulness (không có mệnh đề nội
    dung để kiểm); answer_relevancy vẫn chấm (ragas cho 0 với câu né tránh). Câu từ chối MỘT PHẦN ("…tôi không
    tìm thấy X. Tuy nhiên, tài liệu nêu…" — is_refusal_answer của judge cũ cũng bắt) vẫn chấm đủ, gắn cờ
    partial_refusal. Tổng hợp báo cả "mọi câu" lẫn "bỏ câu từ chối" + tỷ lệ từ chối.
Mặc định chỉ faithfulness + answer_relevancy (user chốt 2026-10-25: context_recall đã đo bằng chỉ số truy xuất IR,
factual_correctness chậm nhất và phạt câu trả lời chi tiết hơn đáp án tham chiếu); --metrics để bật thêm.

Chạy trong venv RIÊNG có `ragas` (xung đột phiên bản với ml_env), KHÔNG import app/:
  <ragas_env>/Scripts/python.exe backend/tests/benchmark/run_ragas_eval.py \\
      --answers reports/ragas_answers.jsonl --out reports/ragas_scores.jsonl [--trace]
Ghi JSONL từng câu, chạy lại bỏ qua câu đã chấm; cuối cùng in + ghi tổng hợp (<out>.summary.json).
"""
import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from importlib.metadata import version

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, FactualCorrectness, Faithfulness

from tests.benchmark.ragas_metrics import is_refusal_answer  # chỉ dùng stdlib, chạy được ngoài ml_env

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "factual_correctness")
DEFAULT_METRICS = ("faithfulness", "answer_relevancy")
JUDGE_MODEL_DEFAULT = "qwen2.5-7b-judge-8k"  # FROM qwen2.5:7b-instruct + PARAMETER num_ctx 8192
EMBEDDING_MODEL_DEFAULT = "BAAI/bge-m3"
REFUSAL_PREFIX = "tôi không tìm thấy thông tin"  # câu mở đầu từ chối của SYSTEM_INSTRUCTION Generator

# Các dạng Gemini dùng (khảo sát 72 câu trả lời): "(Nguồn: x.pdf, Trang 3)", "[Tài liệu 1, tr. 2; Tài liệu 2, tr. 2]",
# "(Trang 4, Tài liệu 2)", "(dựa trên Tài liệu 1…)", lồng 1 cấp "[Nguồn: `x.pdf (Trang 9)`]", và khối cuối
# "**Nguồn trích dẫn:**" + các dòng "- Tài liệu 3, trang 1 (`x.pdf`)".
_SOURCE_LABEL = r"Nguồn(?:\s+(?:trích dẫn|tham khảo))?\s*[*_]*\s*:"
_CITATION_GROUP = re.compile(r"[ \t]*[*_]*[(\[][ \t]*[*_]*[ \t]*(?:" + _SOURCE_LABEL + r"|dựa trên\s+Tài liệu"
                             r"|Tài liệu\s+\d|[Tt]rang\s+\d|tr\.\s*\d)[^()\[\]]*(?:[(\[][^()\[\]]*[)\]][^()\[\]]*)*"
                             r"[)\]][*_]*")
_SOURCE_BLOCK = re.compile(r"(?m)^[ \t*_>-]*" + _SOURCE_LABEL + r".*$"
                           r"(?:\n[ \t]*[-*•][ \t].*(?:Tài liệu|[Tt]rang|\.pdf|\.xlsx|\.csv|\.docx).*$)*")
_EMPTY_LINE = re.compile(r"(?m)^[ \t*_]+$")


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) or os.path.exists(path) else os.path.join(BENCH_DIR, path)


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strip_citations(answer: str) -> tuple[str, int]:
    """Bỏ cụm trích nguồn (xem docstring module). Trả (câu trả lời đã bỏ, số cụm đã bỏ)."""
    text, n_groups = _CITATION_GROUP.subn("", answer)
    text, n_lines = _SOURCE_BLOCK.subn("", text)
    text = _EMPTY_LINE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip(), n_groups + n_lines


def _clean(value) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), 4)


def bootstrap_ci(values: list[float], n_resamples: int = 2000, seed: int = 0) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_resamples))
    return [round(means[int(0.025 * n_resamples)], 3), round(means[int(0.975 * n_resamples) - 1], 3)]


async def score_record(rec: dict, metrics: dict, trace: bool) -> dict:
    response, citations_removed = strip_citations(rec["answer"])
    refusal = response.lower().lstrip(" *_").startswith(REFUSAL_PREFIX)
    out = {"id": rec["id"], "dataset": rec["dataset"], "category": rec["category"], "stratum": rec["stratum"],
           "split": rec["split"], "refusal": refusal,
           "partial_refusal": is_refusal_answer(rec["answer"]) and not refusal, "citations_removed": citations_removed,
           "scores": {}, "seconds": {}, "errors": {}}
    inputs = {
        "faithfulness": dict(user_input=rec["question"], response=response, retrieved_contexts=rec["contexts"]),
        "answer_relevancy": dict(user_input=rec["question"], response=response),
        "context_recall": dict(user_input=rec["question"], retrieved_contexts=rec["contexts"],
                               reference=rec.get("reference_answer")),
        "factual_correctness": dict(response=response, reference=rec.get("reference_answer")),
    }
    for name, metric in metrics.items():
        if name == "faithfulness" and refusal:
            out["skipped"] = "faithfulness: câu từ chối"
            continue
        if name in ("context_recall", "factual_correctness") and not rec.get("reference_answer"):
            out["errors"][name] = "không có reference_answer"
            continue
        if not rec["contexts"] and name in ("faithfulness", "context_recall"):
            out["errors"][name] = "không có context"
            continue
        t0 = time.perf_counter()
        try:
            if name == "faithfulness" and trace:
                # Đúng các bước Faithfulness.ascore (ragas 0.4.3), giữ lại mệnh đề + verdict để kiểm tay
                statements = await metric._create_statements(rec["question"], response)
                verdicts = await metric._create_verdicts(statements, "\n".join(rec["contexts"])) if statements else None
                value = metric._compute_score(verdicts) if verdicts else float("nan")
                out["faithfulness_trace"] = [
                    {"statement": s.statement, "verdict": s.verdict, "reason": s.reason}
                    for s in (verdicts.statements if verdicts else [])]
            else:
                value = (await metric.ascore(**inputs[name])).value
            out["scores"][name] = _clean(value)
        except Exception as exc:  # 1 metric lỗi (vd JSON hỏng) không làm rớt cả câu
            out["errors"][name] = f"{type(exc).__name__}: {str(exc)[:300]}"
        out["seconds"][name] = round(time.perf_counter() - t0, 1)
    return out


def summarize(rows: list[dict], run_config: dict) -> dict:
    def block(group: list[dict]) -> dict:
        res = {"n": len(group), "refusal_rate": round(sum(r["refusal"] for r in group) / len(group), 3),
               "partial_refusal_rate": round(sum(r["partial_refusal"] for r in group) / len(group), 3)}
        for m in run_config["metrics"]:
            for key, subset in ((m, group), (f"{m}_non_refusal", [r for r in group if not r["refusal"]])):
                vals = [r["scores"][m] for r in subset if r["scores"].get(m) is not None]
                res[key] = {"mean": round(sum(vals) / len(vals), 3) if vals else None, "ci95": bootstrap_ci(vals),
                            "n_scored": len(vals)}
        return res

    summary = {"run_config": run_config, "overall": block(rows), "by_dataset": {}, "by_category": {}}
    for field, target in (("dataset", "by_dataset"), ("category", "by_category")):
        groups = defaultdict(list)
        for r in rows:
            groups[r[field]].append(r)
        summary[target] = {k: block(v) for k, v in sorted(groups.items())}
    summary["errors"] = {m: sum(m in r["errors"] for r in rows) for m in run_config["metrics"]}
    return summary


def print_summary(summary: dict):
    cols = [k for m in summary["run_config"]["metrics"] for k in (m, f"{m}_non_refusal")]

    def line(name, b):
        cells = [f"{b[k]['mean']:.3f}" if b[k]["mean"] is not None else "-" for k in cols]
        return f"  {name:<20} n={b['n']:<4} từ chối {b['refusal_rate']:.2f} " + "  ".join(f"{c:>12}" for c in cells)

    labels = [f"{k.split('_')[0][:5]}{'·bỏTC' if k.endswith('_non_refusal') else ''}" for k in cols]
    print(f"\n{'':<44}" + "  ".join(f"{label:>12}" for label in labels))
    print(line("TOÀN BỘ", summary["overall"]))
    for k, b in summary["by_dataset"].items():
        print(line(f"bộ {k}", b))
    for k, b in summary["by_category"].items():
        print(line(k, b))
    print(f"  lỗi theo metric: {summary['errors']}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_DEFAULT)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--ids", help="Chỉ chấm các id này (phân cách dấu phẩy)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trace", action="store_true", help="Lưu mệnh đề + verdict faithfulness để kiểm tay")
    parser.add_argument("--follow", action="store_true",
                        help="Chờ file câu trả lời còn đang được sinh: chấm hết rồi đợi câu mới tới khi đủ --expect")
    parser.add_argument("--expect", type=int, help="Số câu trả lời cần chấm khi dùng --follow")
    args = parser.parse_args()

    selected = [m.strip() for m in args.metrics.split(",") if m.strip()]
    client = AsyncOpenAI(base_url=f"{args.base_url.rstrip('/')}/v1", api_key="ollama", timeout=1800, max_retries=1)
    llm = llm_factory(args.judge_model, client=client, max_tokens=2048)
    embeddings = HuggingFaceEmbeddings(model=args.embedding_model, device="cpu") if "answer_relevancy" in selected else None
    builders = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_recall": lambda: ContextRecall(llm=llm),
        "factual_correctness": lambda: FactualCorrectness(llm=llm),
    }
    metrics = {m: builders[m]() for m in selected}
    out_path = _resolve(args.out)

    def load_records() -> list[dict]:
        records = _read_jsonl(_resolve(args.answers))
        if args.ids:
            wanted = set(args.ids.split(","))
            records = [r for r in records if r["id"] in wanted]
        return records[:args.limit] if args.limit else records

    records = load_records()
    run_config = {"ragas_version": version("ragas"), "judge_model": args.judge_model,
                  "judge_endpoint": "ollama openai-compatible", "embedding_model": args.embedding_model,
                  "metrics": selected, "answers": os.path.basename(args.answers),
                  "preprocessing": "bỏ cụm trích nguồn khỏi câu trả lời; câu MỞ ĐẦU bằng từ chối không chấm "
                                   "faithfulness (từ chối một phần vẫn chấm)",
                  "generator": (records[0].get("run_config") if records else None)}
    print(f"⚙️  {json.dumps(run_config, ensure_ascii=False)}")

    scored = 0
    with open(out_path, "a", encoding="utf-8") as out:
        while True:
            done = {r["id"] for r in _read_jsonl(out_path)}
            todo = [r for r in load_records() if r["id"] not in done]
            if not todo:
                if args.follow and len(done) < (args.expect or 0):
                    await asyncio.sleep(60)
                    continue
                break
            print(f"📋 đã chấm {len(done)}, cần chấm {len(todo)}")
            for rec in todo:
                row = await score_record(rec, metrics, args.trace)
                row["run_config"] = {k: run_config[k] for k in ("ragas_version", "judge_model", "embedding_model")}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                scored += 1
                print(f"[{scored}] {rec['id']} {row['scores']} · {sum(row['seconds'].values()):.0f}s"
                      + (" · từ chối" if row["refusal"] else "")
                      + (f" · lỗi {row['errors']}" if row["errors"] else ""))
            if not args.follow:
                break

    ids = {x["id"] for x in load_records()}
    rows = [r for r in _read_jsonl(out_path) if r["id"] in ids]
    summary = summarize(rows, run_config)
    with open(os.path.splitext(out_path)[0] + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print_summary(summary)


if __name__ == "__main__":
    asyncio.run(main())
