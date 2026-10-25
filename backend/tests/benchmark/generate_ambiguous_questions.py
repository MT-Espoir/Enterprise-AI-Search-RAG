"""
generate_ambiguous_questions.py — Bộ câu hỏi "mơ hồ" 70 câu, CHIA NHÓM (mở rộng bộ 30 câu của
generate_vague_questions.py — 30 câu toàn câu ngắn chưa đủ mạnh để tuyên bố tổng quát):

  broad_short     25  Câu rộng, ngắn, KHÔNG dùng thuật ngữ của tài liệu
  paraphrase      20  Diễn đạt gián tiếp / dùng từ đồng nghĩa
  underspecified  10  Thiếu ngữ cảnh (không nêu tên đối tượng) nhưng vẫn xác định được đối tượng
  multi_doc       15  Cần tổng hợp thông tin từ NHIỀU tài liệu

Sinh DƯ mỗi nhóm (OVERSAMPLE) để người duyệt loại bớt về đúng số mục tiêu (TARGETS).

Chọn nguồn:
  - 3 nhóm 1 tài liệu: mỗi câu 1 TÀI LIỆU KHÁC NHAU, lấy mẫu phân tầng trên TOÀN BỘ tài liệu đủ điều
    kiện (không chỉ 30 tài liệu mẫu) — độ phủ rộng hơn. underspecified ưu tiên tài liệu có đối tượng
    cụ thể (công ty / văn bản pháp luật). Chunk nguồn sạch, KHÔNG thuộc nhãn của bộ 131 câu lẫn bộ 30
    câu mơ hồ.
  - multi_doc: cặp tài liệu liên quan cố định (MULTI_DOC_PAIRS — BCTC 2 năm, riêng vs hợp nhất, 2 công
    ty cùng chỉ số, luật + nghị định hướng dẫn...). Đoạn thứ 2 = đoạn GẦN NGHĨA NHẤT với đoạn 1 (cosine
    embedding) để câu so sánh/tổng hợp tự nhiên. Nhãn required_files = 2 tài liệu nguồn → benchmark đo
    thêm all_required@k (lấy ĐỦ cả 2), không chỉ hit@k.
Pooling như bộ 30 câu: ứng viên từ CẢ snapshot gốc (không header) LẪN snapshot production có header,
judge tính đoạn trả lời được câu hỏi NHƯ ĐÃ ĐẶT.

Kiểm tra chất lượng (cờ cho người duyệt, không tự loại): lần chạy đầu Gemini KHÔNG skip chunk OCR lỗi mà
BỊA nội dung (đọc ngược "000°00S°690°S0Z" thành số tiền) — nên đo (1) chunk nguồn có chữ đọc được không
(tỷ lệ từ là âm tiết phổ biến của corpus), (2) evidence LLM trích có thật nằm trong chunk nguồn không.
--recheck chạy lại 2 kiểm tra này trên ứng viên đã sinh (không gọi LLM).

Dữ liệu GỬI RA NGOÀI: nội dung chunk gửi tới LLM provider (Gemini mặc định, QGEN_PROVIDER).
Chạy:
  python backend/tests/benchmark/generate_ambiguous_questions.py --dry-run
  python backend/tests/benchmark/generate_ambiguous_questions.py
  python backend/tests/benchmark/export_review_sheet.py --set vague70
Chạy lại sẽ TIẾP TỤC từ chỗ dừng.
"""
import argparse
import json
import logging
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import chromadb
import numpy as np

from tests.benchmark.export_review_sheet import SET_TARGETS, SETS
from tests.benchmark.generate_benchmark_questions import (
    HEADER_CHARS, POOL_CHARS, LLMCallFailed, _chunk_ref, _clean_enough, build_pool_search, parse_indices,
)
from tests.benchmark.ragas_metrics import CLAUDE_JUDGE_MODEL_DEFAULT, GEMINI_JUDGE_MODEL_DEFAULT, _call_judge
from tests.benchmark.sample_benchmark_docs import allocate, stratum_of

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BENCH_DIR, "benchmark_chromadb_full")
HEADER_DB_PATH = os.path.join(BENCH_DIR, "benchmark_chromadb_full_prod_header")
COLLECTION_NAME = "benchmark_full"
MANIFEST_PATH = os.path.join(BENCH_DIR, "snapshot_manifest_full.json")
SAMPLE_PATH = os.path.join(BENCH_DIR, "benchmark_sample_full.json")
USED_LABEL_SETS = (SETS["full"]["output"], SETS["vague"]["output"])
OUTPUT_PATH = SETS["vague70"]["candidates"]

SEED = 20261025
TARGETS = SET_TARGETS["vague70"]
OVERSAMPLE = {"broad_short": 3, "paraphrase": 3, "underspecified": 3, "multi_doc": 3}
# underspecified cần đối tượng cụ thể để mô tả gián tiếp (công ty, văn bản pháp luật)
UNDERSPECIFIED_STRATA = {"financial_report": 5, "financial_xlsx": 4, "legal": 4}
GENERATE_MAX_OUTPUT_TOKENS = 1000
MAX_ALTERNATES = 2
# Ngưỡng cờ chất lượng — trung vị chunk của corpus 0.978, p5 0.50 (đo 2026-10-24). Tài liệu tiếng Anh /
# bảng số cũng thấp → chỉ là cờ để người duyệt xem kỹ, không loại tự động
MIN_SOURCE_READABLE = 0.65
MIN_EVIDENCE_SUPPORT = 0.6
COMMON_WORD_MIN_CHUNKS = 50
_SPECIFIC = re.compile(r"\b(19|20)\d{2}\b|hợp nhất|\briêng\b|số\s*\d+/|/\d{4}/|NĐ-CP|QH1\d|TT-BTC|QĐ-", re.IGNORECASE)

# (tài liệu 1, tài liệu 2, quan hệ) — tên file đúng như trong snapshot
MULTI_DOC_PAIRS = [
    ("2016-0100105616-228690-BCTC-vinaconexpdf.pdf", "2017-0100105616-272261-BCTC-vinaconex.pdf", "BCTC hợp nhất Vinaconex 2 năm liền nhau"),
    ("2019-0100105616-414059-BCTC-vinaconex.pdf", "2020-0100105616-20210331-363-BCTC-vinaconex.pdf", "BCTC hợp nhất Vinaconex 2 năm liền nhau"),
    ("2022-0100105616-20230327-1051-BCTC-vinaconex.pdf", "2023-0100105616-20240329-523-BCTC-vinaconex.pdf", "BCTC hợp nhất Vinaconex 2 năm liền nhau"),
    ("2016-0100105616-228721-BCTCR-vinaconex.pdf", "2017-0100105616-272263-BCTCR-vinaconex.pdf", "BCTC riêng Vinaconex 2 năm liền nhau"),
    ("2023-0100105616-20240329-523-BCTC-vinaconex.pdf", "2023-0100105616-20240329-524-BCTCR-vinaconex.pdf", "BCTC hợp nhất và BCTC riêng cùng năm của Vinaconex"),
    ("2022-0102294285-20230330-533-BCTCVinahud.pdf", "BCTCvinahud-31-12-2023.pdf", "BCTC Vinahud 2 năm liền nhau"),
    ("2021-0100105616-20220331-370-BCTC-vinaconex.pdf", "2021-0102294285-20220314-1730-BCTCvinahud.pdf", "BCTC cùng năm của 2 công ty (Vinaconex, Vinahud)"),
    ("Simplize_MWG_FinancialIndicator_20250315.xlsx", "Simplize_FPT_FinancialIndicator_20250315.xlsx", "chỉ số tài chính 2 công ty (MWG, FPT)"),
    ("Simplize_HPG_FinancialIndicator_20250315.xlsx", "Simplize_MSN_FinancialIndicator_20250315.xlsx", "chỉ số tài chính 2 công ty (HPG, MSN)"),
    ("Simplize_VNM_FinancialIndicator_20250315.xlsx", "Simplize_IMP_FinancialIndicator_20250315.xlsx", "chỉ số tài chính 2 công ty (VNM, IMP)"),
    ("Simplize_GAS_PriceHistory_20250315.xlsx", "Simplize_VNM_PriceHistory_20250315.xlsx", "lịch sử giá 2 cổ phiếu (GAS, VNM)"),
    ("luatdoanhnghiep.pdf", "168.pdf", "Luật Doanh nghiệp và nghị định về đăng ký doanh nghiệp"),
    ("45.signed.pdf", "1452020nd-cp_25102023165145.docx", "Bộ luật Lao động và nghị định hướng dẫn"),
    ("38.signed.pdf", "thong-tu-91-2026-tt-btc-quy-dinh-luat-quan-ly-thue-va-hoa-don-dien-tu.pdf", "Luật Quản lý thuế và thông tư hướng dẫn"),
    ("123.signed_01.pdf", "70-nd-cp.signed.pdf", "nghị định hóa đơn chứng từ và nghị định sửa đổi"),
    ("Thong-tu-44-2026-TT-BTC-mau-bao-cao-giam-sat-danh-gia-dau-tu-che-do-bao-cao-truc-tuyen.pdf",
     "Thong-tu-55-2026-TT-BTC-mau-van-ban-bao-cao-lien-quan-den-hoat-dong-dau-tu-tai-Viet-Nam.pdf", "2 thông tư về mẫu báo cáo đầu tư"),
    ("41-2024-qh15.pdf", "45.signed.pdf", "Luật Bảo hiểm xã hội và Bộ luật Lao động"),
    ("mau-dieu-le-cong-ty-co-phan-mau-1.docx", "mau-dieu-le-tnhh-2-thanh-vien-tro-len.docx", "điều lệ mẫu công ty cổ phần và công ty TNHH"),
]

_CORPUS = """Bạn đang xây bộ câu hỏi kiểm thử "người dùng thật" cho hệ thống tìm kiếm tài liệu nội bộ doanh nghiệp (tiếng Việt). Corpus ~90 tài liệu, nhiều tài liệu RẤT GIỐNG NHAU (báo cáo tài chính cùng công ty qua nhiều năm, nghị định sửa đổi lẫn nhau, nhiều mẫu điều lệ, chỉ số tài chính của nhiều công ty)."""

_SINGLE_SOURCE = """TÀI LIỆU: {filename}
PHẦN ĐẦU TÀI LIỆU:
{header}

ĐOẠN NGUỒN:
{chunk}"""

_RULES = {
    "broad_short": """Viết ĐÚNG 1 câu hỏi NGẮN (tối đa khoảng 12 từ), RỘNG, kiểu người dùng gõ vội:
- KHÔNG dùng thuật ngữ chuyên môn xuất hiện trong đoạn nguồn — dùng từ ngữ đời thường thay thế (vd đoạn nói "trợ cấp thôi việc" → hỏi "nghỉ việc được nhận tiền gì").
- KHÔNG nêu năm, loại báo cáo (riêng / hợp nhất), số hiệu, tên văn bản. Được nêu tên ngắn thường gọi của công ty / mã cổ phiếu nếu tự nhiên.
- Câu hỏi về một vấn đề chung, nhưng đoạn nguồn phải chứa câu trả lời trực tiếp.""",
    "paraphrase": """Viết ĐÚNG 1 câu hỏi diễn đạt GIÁN TIẾP hoặc dùng TỪ ĐỒNG NGHĨA:
- Mô tả tình huống / nhu cầu thay vì hỏi thẳng (vd "Nếu công ty chậm trả lương thì nhân viên có đòi được gì không?"), HOẶC thay mọi thuật ngữ chính của đoạn nguồn bằng từ đồng nghĩa / cách nói khác.
- KHÔNG lặp lại cụm từ khóa của đoạn nguồn (trừ tên riêng, con số). Độ dài tự nhiên (8–30 từ).
- KHÔNG nêu số hiệu hay tên đầy đủ văn bản.""",
    "underspecified": """Viết ĐÚNG 1 câu hỏi THIẾU NGỮ CẢNH nhưng VẪN XÁC ĐỊNH ĐƯỢC ĐỐI TƯỢNG:
- KHÔNG nêu tên riêng / mã cổ phiếu của công ty, KHÔNG nêu tên hay số hiệu văn bản chính. Thay bằng mô tả đủ để người hiểu biết nhận ra (vd "tổng công ty xây dựng có trụ sở ở 34 Láng Hạ", "luật về an toàn trên không gian mạng", "hãng sữa lớn nhất Việt Nam").
- KHÔNG nêu năm. Mô tả phải chỉ tới ĐÚNG 1 đối tượng (dùng PHẦN ĐẦU TÀI LIỆU để biết đối tượng là gì).
- Nếu tài liệu không có đối tượng cụ thể nào để mô tả gián tiếp → skip=true.
- Thêm key "entity": tên thật của đối tượng mà câu hỏi ám chỉ.""",
}

_SINGLE_TAIL = """
- Hỏi nội dung thực chất (quy định, điều kiện, con số, thời hạn, trách nhiệm...), không hỏi chữ ký, số trang, mục lục.

Nếu đoạn nguồn lỗi chữ, chỉ toàn bảng số không tiêu đề, hoặc không đủ nội dung → skip=true.

Trả về JSON:
{{"skip": <true|false>, "skip_reason": "<nếu skip>", "question": "<câu hỏi>", "reference_answer": "<câu trả lời ngắn theo đoạn nguồn>", "evidence": "<trích nguyên văn ngắn từ đoạn nguồn>", "entity": "<chỉ nhóm underspecified, còn lại để trống>"}}
"""

_MULTI_PROMPT = _CORPUS + """

QUAN HỆ GIỮA 2 TÀI LIỆU: {relation}

[TÀI LIỆU 1] {filename1}
PHẦN ĐẦU: {header1}
ĐOẠN 1:
{chunk1}

[TÀI LIỆU 2] {filename2}
PHẦN ĐẦU: {header2}
ĐOẠN 2:
{chunk2}

Viết ĐÚNG 1 câu hỏi mà muốn trả lời ĐẦY ĐỦ phải dùng thông tin từ CẢ HAI đoạn (so sánh, đối chiếu thay đổi, tổng hợp, hoặc kết hợp quy định của 2 văn bản) — chỉ đọc 1 đoạn thì KHÔNG trả lời đủ.
- Tự nhiên như người dùng hỏi; được nêu tên ngắn của đối tượng (công ty / chủ đề) và mốc thời gian nếu thật cần để so sánh; KHÔNG nêu số hiệu hay tên đầy đủ văn bản.
- DIỄN ĐẠT LẠI, không chép cụm dài hơn 4 từ liên tiếp (trừ tên riêng, con số).
- Nếu 2 đoạn không đủ liên quan để ghép thành 1 câu hỏi tự nhiên → skip=true.

Trả về JSON:
{{"skip": <true|false>, "skip_reason": "<nếu skip>", "question": "<câu hỏi>", "reference_answer": "<câu trả lời dùng cả 2 đoạn>", "evidence": "<trích ngắn từ đoạn 1> || <trích ngắn từ đoạn 2>"}}
"""

_POOL_PROMPT = """Bạn chấm nhãn cho bộ câu hỏi tìm kiếm của người dùng thật (câu có thể mơ hồ, gián tiếp, không nêu năm / số hiệu văn bản). Với CÂU HỎI dưới đây, xét từng ĐOẠN và cho biết đoạn nào chứa thông tin TRỰC TIẾP trả lời câu hỏi NHƯ ĐÃ ĐẶT — toàn bộ hoặc một phần chính của câu trả lời{multi_note}.

Câu hỏi không nêu năm / loại báo cáo / văn bản cụ thể thì đoạn của tài liệu khác (năm khác, bản riêng hay hợp nhất, văn bản tương tự) VẪN TÍNH nếu trả lời đúng điều được hỏi cho đúng đối tượng được hỏi, dù con số khác đáp án tham chiếu. Đoạn chỉ cùng chủ đề, nhắc khái niệm nhưng không trả lời thì KHÔNG tính.

CÂU HỎI: {question}
ĐÁP ÁN THAM CHIẾU (chỉ để tham khảo): {answer}

CÁC ĐOẠN:
{candidates}

Trả về JSON: {{"relevant": [<danh sách số thứ tự các đoạn trả lời được>]}}
"""


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", (text or "").lower()))


def source_readability(text: str, chunk_freq: Counter) -> float:
    # Tỷ lệ từ (bỏ số) là từ phổ biến của corpus — chunk OCR đảo ngược ("UbY Iep AeA") rất thấp
    words = [w for w in _words(text) if not w.isdigit()]
    return sum(chunk_freq[w] >= COMMON_WORD_MIN_CHUNKS for w in words) / len(words) if words else 0.0


def evidence_support(evidence: str, texts: list[str], multi: bool) -> float:
    # Tỷ lệ từ của evidence có trong chunk nguồn (multi_doc: từng vế "a || b" so với chunk tương ứng)
    evidence = re.sub(r"^\[Đối tượng:[^\]]*\]\s*", "", evidence or "")
    parts = evidence.split("||") if multi else [evidence]
    scores = []
    for i, part in enumerate(parts):
        source = set(_words(texts[min(i, len(texts) - 1)] if multi else " ".join(texts)))
        words = [w for w in _words(part) if len(w) > 1]
        scores.append(sum(w in source for w in words) / len(words) if words else 0.0)
    return min(scores)


def quality_flags(candidate: dict, texts: list[str], chunk_freq: Counter) -> list[str]:
    flags = []
    if min(source_readability(t, chunk_freq) for t in texts) < MIN_SOURCE_READABLE:
        flags.append("source_text_garbled")
    if evidence_support(candidate.get("evidence"), texts, candidate["category"] == "multi_doc") < MIN_EVIDENCE_SUPPORT:
        flags.append("evidence_not_in_source")
    return flags


def corpus_chunk_freq(documents: list[str]) -> Counter:
    return Counter(w for d in documents for w in set(_words(d)))


def recheck(collection):
    # Chạy lại cờ chất lượng trên ứng viên đã sinh — không gọi LLM, không đổi trạng thái duyệt
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        output = json.load(f)
    data = collection.get(include=["documents"])
    text_of = dict(zip(data["ids"], data["documents"]))
    freq = corpus_chunk_freq(data["documents"])
    added = Counter()
    for c in output["candidates"]:
        new = quality_flags(c, [text_of[s["chunk_id"]] for s in c["source_chunks"]], freq)
        c["flags"] = [f for f in c["flags"] if f not in QUALITY_FLAGS] + new
        added.update(new)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"🔎 Cờ chất lượng trên {len(output['candidates'])} ứng viên: {dict(added)} → {OUTPUT_PATH}")


QUALITY_FLAGS = ("source_text_garbled", "evidence_not_in_source")


def _eligible_docs(manifest: dict, sample: dict) -> list[dict]:
    excluded = {d["filename"] for d in sample["excluded_documents"]}
    return [{**d, "stratum": stratum_of(d["filename"]), "ext": os.path.splitext(d["filename"])[1].lower()}
            for d in manifest["documents"] if d["filename"] not in excluded]


def plan_tasks(docs: list[dict], by_doc: dict[str, list[dict]], used: set[str], embedding_of) -> list[dict]:
    """Kế hoạch sinh câu hỏi — hàm thuần theo SEED (dry-run và chạy thật ra cùng kế hoạch)."""
    rng = random.Random(SEED)
    usable = {d["doc_id"]: [c for c in by_doc[d["doc_id"]] if c["id"] not in used and _clean_enough(c["text"], d["ext"])]
              for d in docs}
    by_stratum = defaultdict(list)
    for d in sorted(docs, key=lambda d: d["filename"]):
        if usable[d["doc_id"]]:
            by_stratum[d["stratum"]].append(d)

    n_single = sum(TARGETS[c] + OVERSAMPLE[c] for c in ("broad_short", "paraphrase", "underspecified"))
    alloc = allocate({s: len(v) for s, v in by_stratum.items()}, n_single)
    picked = {s: rng.sample(by_stratum[s], alloc[s]) for s in sorted(alloc)}

    def single(category: str, d: dict) -> dict:
        # chunk dự phòng CÙNG tài liệu: LLM skip (OCR lỗi, bảng trống) thì thử tiếp, giữ nguyên phân bổ tài liệu
        options = rng.sample(usable[d["doc_id"]], min(1 + MAX_ALTERNATES, len(usable[d["doc_id"]])))
        return {"category": category, "doc": d, "chunks": options[:1], "alternates": options[1:]}

    tasks = []
    for stratum, n in UNDERSPECIFIED_STRATA.items():
        for d in picked[stratum][:n]:
            tasks.append(single("underspecified", d))
        picked[stratum] = picked[stratum][n:]
    # Xen kẽ các nhóm tài liệu rồi chia broad_short / paraphrase theo tỉ lệ → cả 2 nhóm câu đều phủ mọi loại tài liệu
    rest, queues = [], [list(picked[s]) for s in sorted(picked)]
    while any(queues):
        for q in queues:
            if q:
                rest.append(q.pop(0))
    n_short = TARGETS["broad_short"] + OVERSAMPLE["broad_short"]
    for i, d in enumerate(rest):
        short_so_far = sum(t["category"] == "broad_short" for t in tasks)
        category = "broad_short" if short_so_far < round((i + 1) * n_short / len(rest)) else "paraphrase"
        tasks.append(single(category, d))

    doc_of = {d["filename"]: d for d in docs}
    for f1, f2, relation in MULTI_DOC_PAIRS[:TARGETS["multi_doc"] + OVERSAMPLE["multi_doc"]]:
        d1, d2 = doc_of[f1], doc_of[f2]
        c1 = rng.choice(usable[d1["doc_id"]])
        # Đoạn 2 = đoạn gần nghĩa nhất với đoạn 1 trong tài liệu 2
        candidates = usable[d2["doc_id"]]
        v1 = embedding_of(c1["id"])
        c2 = max(candidates, key=lambda c: float(np.dot(v1, embedding_of(c["id"]))))
        tasks.append({"category": "multi_doc", "doc": d1, "doc2": d2, "relation": relation, "chunks": [c1, c2]})
    return tasks


def _pool(question: str, answer: str, sources: list[dict], pool: list[dict], multi: bool,
          provider: str, api_key: str, model: str) -> tuple[list[dict], set[str], int]:
    source_ids = {s["id"] for s in sources}
    candidates = list(sources) + [c for c in pool if c["id"] not in source_ids]
    random.Random(question).shuffle(candidates)
    listing = "\n\n".join(
        f"[{i + 1}] (tài liệu: {c['meta']['filename']})\n{c['text'][:POOL_CHARS]}" for i, c in enumerate(candidates))
    result = _call_judge(provider, api_key, model, _POOL_PROMPT.format(
        question=question, answer=answer, candidates=listing,
        multi_note=" (câu hỏi cần ghép thông tin từ nhiều tài liệu: đoạn chứa MỘT PHẦN cần thiết cũng tính)" if multi else "",
    ))
    if not isinstance(result.get("relevant"), list):
        raise LLMCallFailed(result.get("reasoning") or "pooling không trả về danh sách 'relevant'")
    picked = {candidates[idx - 1]["id"] for idx in parse_indices(result["relevant"], len(candidates))}
    others = [c for c in candidates if c["id"] in picked and c["id"] not in source_ids]
    return others, picked & source_ids, len(candidates)


def run(dry_run: bool, recheck_only: bool = False):
    from app.core import context_header

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        sample = json.load(f)
    used: set[str] = set()
    for path in USED_LABEL_SETS:
        with open(path, encoding="utf-8") as f:
            dataset = json.load(f)
        if dataset["snapshot_fingerprint"] != manifest["fingerprint"]:
            sys.exit(f"❌ {os.path.basename(path)} không cùng fingerprint với snapshot.")
        used |= {cid for q in dataset["questions"] for cid in q["relevant_chunk_ids"]}

    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    if recheck_only:
        recheck(collection)
        return
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    by_doc: dict[str, list[dict]] = defaultdict(list)
    emb = {}
    for cid, text, meta, vec in zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"]):
        by_doc[meta["doc_id"]].append({"id": cid, "text": text, "meta": meta})
        v = np.asarray(vec, dtype=float)
        emb[cid] = v / (np.linalg.norm(v) or 1.0)
    for chunks in by_doc.values():
        chunks.sort(key=lambda c: (c["meta"].get("page_num") or 0, c["meta"].get("chunk_index") or 0))

    docs = _eligible_docs(manifest, sample)
    tasks = plan_tasks(docs, by_doc, used, emb.__getitem__)
    print(f"📋 {len(tasks)} việc ({dict(Counter(t['category'] for t in tasks))}) trên "
          f"{len({t['doc']['doc_id'] for t in tasks})} tài liệu — bỏ {len(used)} chunk đã là nhãn của 2 bộ trước\n")
    if dry_run:
        for t in tasks:
            name = t["doc"]["filename"] + (f"  +  {t['doc2']['filename']}" if t["category"] == "multi_doc" else "")
            preview = re.sub(r"\s+", " ", t["chunks"][0]["text"])[:90]
            print(f"  [{t['category']:<14}|{t['doc']['stratum']:<16}] {name[:70]:<70} | {preview}")
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
                  "generator": {"provider": provider, "model": model, "seed": SEED, "kind": "vague70",
                                "targets": TARGETS, "oversample": OVERSAMPLE},
                  "candidates": [], "skipped": []}
    produced = {c["task_key"] for c in output["candidates"]}
    skipped = {s["task_key"] for s in output["skipped"]}

    def save():
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    context_header.set_enabled(True)  # trước khi dựng BM25 của snapshot có header (xem generate_vague_questions.py)
    search_base = build_pool_search(collection)
    search_header = build_pool_search(chromadb.PersistentClient(path=HEADER_DB_PATH).get_collection(COLLECTION_NAME))
    group_files = {f for g in sample["equivalence_groups"] for f in g}
    equivalents = {f: [x for g in sample["equivalence_groups"] if f in g for x in g] for f in group_files}
    head_of = lambda d: by_doc[d["doc_id"]][0]["text"][:HEADER_CHARS]
    chunk_freq = corpus_chunk_freq(data["documents"])

    def attempt(i: int, task: dict, sources: list[dict], key: str) -> str:
        """1 lần sinh câu hỏi từ sources → "ok" | "skip" (LLM từ chối chunk) | "fail" (lỗi gọi LLM)."""
        category, doc = task["category"], task["doc"]
        multi = category == "multi_doc"
        label = doc["filename"] + (f" + {task['doc2']['filename']}" if multi else "")
        print(f"[{i}/{len(tasks)}] {category} — {label}" + ("" if sources is task["chunks"] else "  (chunk dự phòng)"))
        if multi:
            prompt = _MULTI_PROMPT.format(relation=task["relation"], filename1=doc["filename"], header1=head_of(doc),
                                          chunk1=sources[0]["text"], filename2=task["doc2"]["filename"],
                                          header2=head_of(task["doc2"]), chunk2=sources[1]["text"])
        else:
            prompt = (_CORPUS + "\n\n" + _SINGLE_SOURCE.format(filename=doc["filename"], header=head_of(doc),
                                                               chunk=sources[0]["text"])
                      + "\n\n" + _RULES[category] + _SINGLE_TAIL)
        gen = _call_judge(provider, api_key, model, prompt, max_output_tokens=GENERATE_MAX_OUTPUT_TOKENS)
        if "skip" not in gen and "question" not in gen:
            print(f"   ❌ lỗi gọi LLM, để lần chạy sau làm lại: {gen.get('reasoning')}")
            return "fail"
        if gen.get("skip") or not gen.get("question"):
            reason = gen.get("skip_reason") or "không có câu hỏi"
            print(f"   ⏭  skip: {reason}")
            output["skipped"].append({"task_key": key, "category": category, "filename": label, "reason": reason})
            save()
            return "skip"

        question = gen["question"].strip()
        pool = {c["id"]: c for c in search_base(question) + search_header(question)}
        try:
            pooled, confirmed, pool_size = _pool(question, gen.get("reference_answer", ""), sources,
                                                 list(pool.values()), multi, provider, api_key, model)
        except LLMCallFailed as exc:
            print(f"   ❌ lỗi pooling, để lần chạy sau làm lại: {exc}")
            return "fail"
        source_refs = [_chunk_ref(c) for c in sources]
        relevant = [{**r, "via": "source"} for r in source_refs] + [{**_chunk_ref(c), "via": "pooled"} for c in pooled]
        source_files = [r["filename"] for r in source_refs]
        relevant_files = sorted({r["filename"] for r in relevant} | {e for f in source_files for e in equivalents.get(f, [])})
        flags = []
        if len(confirmed) < len(sources):
            flags.append("source_not_confirmed")
        if any(r["filename"] not in source_files for r in relevant):
            flags.append("answer_also_in_other_document")
        if not multi and _SPECIFIC.search(question):
            flags.append("mentions_year_type_or_number")
        flags += quality_flags({"evidence": gen.get("evidence"), "category": category}, [c["text"] for c in sources], chunk_freq)
        entity = str(gen.get("entity") or "").strip()
        output["candidates"].append({
            "id": f"a{len(output['candidates']) + 1:03d}",
            "task_key": key,
            "question": question,
            "reference_answer": gen.get("reference_answer"),
            "evidence": (f"[Đối tượng: {entity}] " if category == "underspecified" and entity else "") + (gen.get("evidence") or ""),
            "category": category,
            "stratum": doc["stratum"],
            "source_chunks": source_refs,
            "relevant_chunks": relevant,
            "expected_sources": [{"filename": f, "page_num": None} for f in relevant_files],
            # multi_doc: PHẢI lấy đủ các tài liệu này mới trả lời đủ (benchmark đo all_required@k)
            **({"required_files": source_files, "relation": task["relation"]} if multi else {}),
            "pooling": {"pool_size": pool_size, "sources_confirmed": len(confirmed), "added_chunks": len(pooled),
                        "systems": ["baseline_no_header", "prod_header"]},
            "flags": flags,
            "review": {"status": "pending", "note": ""},
        })
        save()
        print(f"   ✅ {question}")
        print(f"      relevant: {len(relevant)} chunk / {len(relevant_files)} tài liệu"
              + (f"  ⚠ {', '.join(flags)}" if flags else ""))
        return "ok"

    failed = 0
    for i, task in enumerate(tasks, 1):
        tries = [task["chunks"]] + [[c] for c in task.get("alternates", [])]
        keys = [f"{task['category']}:" + "+".join(c["id"] for c in t) for t in tries]
        if any(k in produced for k in keys):
            continue  # việc đã xong ở lần chạy trước
        for sources, key in zip(tries, keys):
            if key in skipped:
                continue
            result = attempt(i, task, sources, key)
            if result == "fail":
                failed += 1
            if result != "skip":
                break

    counts = Counter(c["category"] for c in output["candidates"])
    print(f"\n💾 {len(output['candidates'])} ứng viên {dict(counts)} (mục tiêu {TARGETS}), "
          f"{len(output['skipped'])} skip → {OUTPUT_PATH}")
    short = {c: TARGETS[c] - counts.get(c, 0) for c in TARGETS if counts.get(c, 0) < TARGETS[c]}
    if short:
        print(f"⚠️  Nhóm THIẾU so với mục tiêu: {short} — cần bổ sung trước khi duyệt")
    if failed:
        print(f"⚠️  {failed} việc lỗi gọi LLM (chưa lưu) — chạy lại lệnh để làm tiếp phần còn thiếu.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch, không gọi LLM")
    parser.add_argument("--recheck", action="store_true", help="Chỉ chạy lại cờ chất lượng trên ứng viên đã sinh")
    args = parser.parse_args()
    run(dry_run=args.dry_run, recheck_only=args.recheck)
