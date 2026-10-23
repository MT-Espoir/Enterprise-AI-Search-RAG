"""Gợi ý header ngữ cảnh cho 1 tài liệu (S2, contextual_header_rollout_plan.md §2b).

Thứ tự nguồn:
  1. RULE (không gọi LLM) — nhận diện theo mẫu đã quan sát trên 89 tài liệu thật:
       - bảng Excel có dòng "| Tên công ty | ... |" + "| Tên báo cáo | ... |" (Simplize)
       - báo cáo tài chính: loại riêng/hợp nhất + năm + công ty LẬP báo cáo
       - văn bản pháp luật: loại + số hiệu CỦA CHÍNH văn bản + trích yếu, "bản hợp nhất"
  2. SLM local (Ollama, dữ liệu không ra khỏi máy — quyết định D2) khi rule không đủ
     trường: SLM chỉ TRÍCH các trường, code tự ráp câu (không để SLM viết header tự do).
  3. Không đủ thông tin → header=None, needs_review=True.

Header chỉ chứa thông tin NHẬN DIỆN (loại / số hiệu / năm / đơn vị), không tóm tắt nội
dung — nó được lặp lại ở MỌI chunk của tài liệu khi embed.
"""
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

MAX_HEADER_CHARS = 250
OPENING_CHUNKS = 6          # số chunk đầu (theo trang, chunk_index) dùng làm "phần mở đầu"
SLM_OPENING_CHARS = 1800

SOURCE_RULE = "rule"
SOURCE_SLM = "slm"


@dataclass
class HeaderSuggestion:
    header: str | None
    source: str | None              # "rule" | "slm" | None
    kind: str                       # excel_table | financial_report | legal | other
    needs_review: bool = False
    fields: dict = field(default_factory=dict)


# ── Chuẩn hóa / kiểm tra header ─────────────────────────────────────────────

def normalize_header(raw: str | None) -> str | None:
    """1 dòng, bọc trong [...], ≤ MAX_HEADER_CHARS. Rỗng → None. Quá dài → ValueError."""
    if raw is None:
        return None
    text = re.sub(r"\s+", " ", str(raw)).strip()
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    text = text.strip()
    if not text:
        return None
    header = f"[{text}]"
    if len(header) > MAX_HEADER_CHARS:
        raise ValueError(f"Header dài {len(header)} ký tự, tối đa {MAX_HEADER_CHARS}.")
    return header


# Nguyên âm tiếng Việt có dấu (KHÔNG gồm 'đ' — phụ âm, không bao giờ nằm giữa âm tiết bị tách)
_ACCENTED_VOWELS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ")
_ACCENTED_VOWELS |= {c.upper() for c in _ACCENTED_VOWELS}


def repair_split_syllables(text: str) -> str:
    """Ghép âm tiết bị tách do chữ ký số / font PDF: 'T Ổ NG CÔNG TY C Ổ PH Ầ N' → 'TỔNG CÔNG TY CỔ PHẦN'.

    Token đúng 1 NGUYÊN ÂM có dấu được ghép với token ngay trước (≤ 2 chữ) và với
    token ngay sau (≤ 2 chữ) — trừ khi token sau lại đứng trước 1 ký tự có dấu khác (nó thuộc âm
    tiết kế tiếp, vd 'PH' trong 'C Ổ PH Ầ N'). Chỉ dùng cho đầu vào gợi ý, KHÔNG sửa chunk lưu trữ.
    """
    tokens = text.split(" ")
    is_single = [len(t) == 1 and t in _ACCENTED_VOWELS for t in tokens]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if is_single[i] and out and 0 < len(out[-1]) <= 2 and out[-1].isalpha():
            merged = out.pop() + tokens[i]
            nxt = i + 1
            if (nxt < len(tokens) and 0 < len(tokens[nxt]) <= 2 and tokens[nxt].isalpha()
                    and not (nxt + 1 < len(tokens) and is_single[nxt + 1])):
                merged += tokens[nxt]
                i += 1
            out.append(merged)
        else:
            out.append(tokens[i])
        i += 1
    return " ".join(out)


def _clip(text: str, limit: int) -> str:
    text = text.strip(" ,;:-–")
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-–") + "…"


def _mostly_upper(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


def _sentence_case(text: str) -> str:
    """'LUẬT AN NINH MẠNG' → 'Luật an ninh mạng' (chỉ khi gần như toàn chữ hoa)."""
    if _mostly_upper(text):
        text = text.lower()
        return text[:1].upper() + text[1:]
    return text


# ── Rule 1: bảng Excel có dòng thông tin (Simplize) ─────────────────────────

def _table_field(text: str, label: str) -> str | None:
    m = re.search(r"\|\s*" + re.escape(label) + r"\s*\|\s*([^|]{2,120}?)\s*\|", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _rule_excel(filename: str, opening: str) -> HeaderSuggestion | None:
    company = _table_field(opening, "Tên công ty")
    report = _table_field(opening, "Tên báo cáo")
    if not (company and report):
        return None
    ticker = re.match(r"(?i)simplize_([A-Z0-9]{2,5})_", filename)
    ticker_info = f" (mã CK: {ticker.group(1).upper()})" if ticker else ""
    source = " — nguồn Simplize" if filename.lower().startswith("simplize_") else ""
    header = f"[{_sentence_case(report)} của {company}{ticker_info}{source}]"
    return HeaderSuggestion(header, SOURCE_RULE, "excel_table", fields={"company": company, "report": report})


# ── Rule 2: báo cáo tài chính ───────────────────────────────────────────────

_COMPANY_TAIL = r"[^,.;:()|\n\"“”]{3,90}?"
_COMPANY_STOP = (r"(?=\s*(?:\(|,|\.|;|:|\"|“|”|\||$)|\s(?:và các|tại ngày|tại|cho năm|cho|được|đã|theo|ngày|năm|"
                 r"bao gồm|gọi tắt|Ký|Signature|Digitally|Date|Reason|DN|CN|OU|O=|THUYẾT MINH|THUYET MINH|"
                 r"BÁO CÁO|BAO CAO)\b)")
_COMPANY_HEAD = r"(?:Tổng\s+Công\s+ty|Công\s+ty)\s+(?:Cổ\s+phần|CP|TNHH)\s+"
_COMPANY = re.compile("(" + _COMPANY_HEAD + _COMPANY_TAIL + ")" + _COMPANY_STOP, re.IGNORECASE)
# Đơn vị KÝ SỐ báo cáo = đơn vị lập báo cáo (tín hiệu mạnh nhất, nằm ngay trang đầu)
_SIGNER_COMPANY = re.compile(
    r"(?:Digitally signed by|Ký bởi:?)\s+(" + _COMPANY_HEAD + _COMPANY_TAIL + ")" + _COMPANY_STOP, re.IGNORECASE)
# "Báo cáo tài chính ... của {công ty}" — ở phần mở đầu là công ty lập báo cáo; ở thuyết minh có thể là công ty con
_REPORTING_COMPANY = re.compile(
    r"báo cáo tài chính[^.]{0,40}?của\s+(" + _COMPANY_HEAD + _COMPANY_TAIL + ")" + _COMPANY_STOP, re.IGNORECASE)
_YEAR_END = re.compile(r"(?:kết thúc|tại)\s+ngày\s+31\s*(?:tháng\s*12|/\s*12)\s*(?:năm|/)\s*(\d{4})", re.IGNORECASE)
_FILENAME_STOPWORDS = {"signed", "bctc", "bctcr", "copy", "final", "report"}
_SMALL_WORDS = {"và", "của", "tại", "các", "một"}
_COMPANY_PREFIX_CASE = {"tổng": "Tổng", "công": "Công", "ty": "ty", "cổ": "Cổ", "phần": "phần", "tnhh": "TNHH", "cp": "CP"}


def _normalize_company(name: str) -> str:
    """Gộp khoảng trắng; tên toàn chữ HOA → 'Tổng Công ty Cổ phần Xuất Nhập Khẩu và Xây Dựng Việt Nam'."""
    name = re.sub(r"\s+", " ", name).strip()
    if not _mostly_upper(name):
        return re.sub(r"^(Tổng\s+)?công\s+ty", lambda m: (m.group(1) or "") + "Công ty", name, flags=re.IGNORECASE)
    out = []
    for i, word in enumerate(name.lower().split(" ")):
        if i < 4 and word in _COMPANY_PREFIX_CASE:
            out.append(_COMPANY_PREFIX_CASE[word])
        elif word in _SMALL_WORDS:
            out.append(word)
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


def _alias_from_filename(filename: str, full_text: str) -> str | None:
    """Tên thương hiệu xuất hiện cả ở tên file lẫn nội dung (vd 'vinaconex')."""
    for token in re.findall(r"[A-Za-z]{5,}", filename):
        if token.lower() in _FILENAME_STOPWORDS:
            continue
        variants = Counter(re.findall(rf"\b{token}\b", full_text, re.IGNORECASE))
        if sum(variants.values()) >= 3:
            best = variants.most_common(1)[0][0]
            return best[:1].upper() + best[1:].lower() if best.isupper() else best
    return None


def _report_year(filename: str, full_text: str) -> str | None:
    m = re.match(r"((?:19|20)\d{2})-", filename) or re.search(r"\d{2}-\d{2}-((?:19|20)\d{2})", filename)
    if m:
        return m.group(1)
    years = Counter(_YEAR_END.findall(full_text))
    if not years:
        return None
    top = max(years.values())
    return max(y for y, c in years.items() if c == top)  # hòa → năm lớn hơn (năm báo cáo, không phải cột so sánh)


def _rule_financial(filename: str, opening: str, full_text: str) -> HeaderSuggestion | None:
    lower_name = filename.lower()
    if "bctc" not in lower_name and "báo cáo tài chính" not in opening.lower():
        return None
    year = _report_year(filename, full_text)
    if not year:
        return None

    if "bctcr" in lower_name:
        kind = "riêng"
    else:
        low = full_text.lower()
        consolidated = low.count("báo cáo tài chính hợp nhất")
        separate = low.count("báo cáo tài chính riêng")
        kind = "hợp nhất" if consolidated > separate else "riêng" if separate > consolidated else None

    company = None
    # Thứ tự tin cậy: đơn vị ký số → "BCTC của X" ở phần mở đầu → "BCTC của X" toàn văn → công ty ở phần mở đầu
    for pattern, text in ((_SIGNER_COMPANY, opening), (_REPORTING_COMPANY, opening),
                          (_REPORTING_COMPANY, full_text), (_COMPANY, opening)):
        found = Counter(_normalize_company(c) for c in pattern.findall(text))
        if found:
            company = found.most_common(1)[0][0]
            break
    alias = _alias_from_filename(filename, full_text)
    if company and alias and alias.lower() not in company.lower():
        company = f"{company} ({alias})"
    company = company or alias
    if not company:
        return None

    kind_text = f" {kind}" if kind else ""
    header = f"[Báo cáo tài chính{kind_text} năm {year} của {company}, năm tài chính kết thúc ngày 31/12/{year}]"
    return HeaderSuggestion(header, SOURCE_RULE, "financial_report",
                            needs_review=kind is None or company == alias,
                            fields={"year": year, "report_kind": kind, "company": company})


# ── Rule 3: văn bản pháp luật ───────────────────────────────────────────────

_NUMBER = r"(\d{1,4}/\d{4}/(?:NĐ-CP|QH\d{1,2}|TT-[A-ZĐ]{2,10}|QĐ-[A-ZĐ]{2,10}|NQ-[A-ZĐ]{2,10}))"
_DOC_NUMBER = re.compile(r"\b" + _NUMBER, re.IGNORECASE)
# Số hiệu CỦA CHÍNH văn bản: "Số: 145/2020/NĐ-CP", "Luật số: 24/2018/QH14" (có dấu hai chấm) —
# khác "Căn cứ Nghị định số 171/2016/NĐ-CP" (trích dẫn văn bản khác, không có dấu hai chấm)
_OWN_NUMBER = re.compile(r"(?:^|\s)(?:Luật\s+)?[Ss]ố\s*[:：]\s*" + _NUMBER)
# Dòng đầu bản hợp nhất: "Bộ luật Lao động số 45/2019/QH14 ngày 20 tháng 11 năm 2019 của Quốc hội, có hiệu lực..."
_CONSOLIDATED_NUMBER = re.compile(r"số\s+" + _NUMBER + r"\s+ngày[^.]{0,80}?của[^.]{0,30}?có hiệu lực", re.IGNORECASE)
_KIND_BY_SUFFIX = (("NĐ-CP", "Nghị định"), ("QH", "Luật"), ("TT-", "Thông tư"), ("QĐ-", "Quyết định"), ("NQ-", "Nghị quyết"))
_KIND_WORDS = ("BỘ LUẬT", "LUẬT", "NGHỊ ĐỊNH", "THÔNG TƯ", "QUYẾT ĐỊNH", "NGHỊ QUYẾT")
_MIXED_TITLE_AFTER_KIND = re.compile(
    r"\b(?:BỘ LUẬT|LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH|NGHỊ QUYẾT)\s+([A-ZÀ-Ỹ][^\n]{4,170}?)"
    r"(?=\s+(?:Căn cứ|CHƯƠNG|Chương|Điều 1|Theo đề nghị))")
_ISSUED_AS = re.compile(r"ban hành\s+(?:Bộ luật|Luật|Nghị định|Thông tư|Quyết định|Nghị quyết)\s+([^.;]{5,160})",
                        re.IGNORECASE)


def _own_number(filename: str, opening: str, full_text: str) -> str | None:
    m = _OWN_NUMBER.search(opening) or _CONSOLIDATED_NUMBER.search(opening[:600])
    if m:
        return m.group(1)
    # Bản ký số bắt đầu giữa văn bản (vd "70-nd-cp.signed.pdf"): số hiệu khớp số ở đầu tên file
    lead = re.match(r"(\d{1,4})\D", filename) or re.search(r"-(\d{1,4})-(?:19|20)\d{2}", filename)
    if lead:
        hits = Counter(n for n in _DOC_NUMBER.findall(full_text) if n.split("/")[0] == lead.group(1))
        if hits:
            return hits.most_common(1)[0][0]
    return None


def _uppercase_title(opening: str, kind: str) -> str | None:
    """Cụm CHỮ HOA ngay sau loại văn bản: 'NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT ... Căn cứ' → 'QUY ĐỊNH CHI TIẾT ...'.
    Tìm đúng loại của số hiệu TRƯỚC — tránh bắt 'LUẬT' nằm trong trích yếu '... CỦA BỘ LUẬT LAO ĐỘNG' của 1 Nghị định."""
    own = kind.upper()
    for kind in (own,) + tuple(k for k in _KIND_WORDS if k != own):
        m = re.search(rf"\b{kind}\b", opening)
        if not m:
            continue
        words = []
        for token in opening[m.end():].split():
            letters = [c for c in token if c.isalpha()]
            if letters and not all(c.isupper() for c in letters):
                break
            words.append(token)
            if len(words) >= 35:
                break
        if len(words) >= 2 or (words and len(words[0]) > 3):
            return " ".join(words)
    return None


def _kind_of(number: str, opening: str) -> str:
    for suffix, kind in _KIND_BY_SUFFIX:
        if suffix.lower() in number.lower():
            if kind == "Luật" and re.search(r"\bBỘ LUẬT\b", opening):
                return "Bộ luật"
            return kind
    return "Văn bản"


def _rule_legal(filename: str, opening: str, full_text: str) -> HeaderSuggestion | None:
    number = _own_number(filename, opening, full_text)
    if not number:
        return None
    number = re.sub(r"(?i)nđ-cp", "NĐ-CP", number).upper()
    kind = _kind_of(number, opening)

    title = _uppercase_title(opening, kind)
    if title:
        title = _sentence_case(title)
    else:
        m = _MIXED_TITLE_AFTER_KIND.search(opening) or _ISSUED_AS.search(full_text[:6000])
        title = m.group(1).strip() if m else None
    consolidated = bool(re.search(r"được sửa đổi,\s*bổ sung bởi", opening[:800]))

    if title and kind in ("Luật", "Bộ luật") and len(title.split()) <= 8:
        # Tên luật ngắn: "Luật An ninh mạng số 24/2018/QH14"
        header = f"[{kind} {title[:1].upper() + title[1:]} số {number}"
    else:
        header = f"[{kind} số {number}"
        if title:
            header += " " + _clip(title[:1].lower() + title[1:], 150)
    header += (" (bản hợp nhất)" if consolidated else "") + "]"
    return HeaderSuggestion(header, SOURCE_RULE, "legal", needs_review=title is None,
                            fields={"kind": kind, "number": number, "title": title, "consolidated": consolidated})


# ── SLM fallback ─────────────────────────────────────────────────────────────

# KHÔNG liệt kê ví dụ cho "loai": dry-run 2026-10-23 trên 36 tài liệu thật, qwen2.5:3b chép ví dụ
# đầu tiên ("Quy trình") cho cả luật/nghị định/báo cáo thường niên.
SLM_SYSTEM_INSTRUCTION = (
    "Bạn trích thông tin NHẬN DIỆN của một tài liệu từ phần mở đầu của nó. Chỉ dùng thông tin có "
    "trong văn bản, không bịa, không đoán. Trả về JSON với các key: "
    '"ten" (tiêu đề tài liệu đúng như văn bản ghi, ngắn gọn), '
    '"loai" (loại tài liệu đúng như chính văn bản tự gọi trong tiêu đề; rỗng nếu tiêu đề không nói rõ), '
    '"don_vi" (tổ chức hoặc tác giả ban hành; rỗng nếu không rõ), '
    '"nam" (năm ban hành/xuất bản; rỗng nếu không rõ). Không thêm key khác.'
)
_EMPTY_VALUES = {"không rõ", "không xác định", "chưa rõ", "unknown", "n/a", "none", "null", "..."}
_DOC_TYPE_WORDS = ("luật", "bộ luật", "nghị định", "thông tư", "quyết định", "nghị quyết", "quy chế", "quy trình",
                   "quy định", "mẫu", "điều lệ", "hợp đồng", "báo cáo", "thỏa ước", "dự thảo", "bài tập", "tài liệu")


def _slm_suggest(filename: str, opening: str, base_url: str, model_name: str, timeout: int) -> HeaderSuggestion | None:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SLM_SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Tên file: {filename}\n\nPhần mở đầu:\n{opening[:SLM_OPENING_CHARS]}"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    try:
        response = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        data = json.loads(response.json().get("message", {}).get("content", "") or "{}")
    except Exception as exc:
        logger.warning(f"Gợi ý header bằng SLM thất bại cho '{filename}' ({exc}) — để người dùng tự nhập.")
        return None

    def get(key: str) -> str:
        value = re.sub(r"\s+", " ", str(data.get(key) or "")).strip(" .,;:-–()")
        return "" if value.lower() in _EMPTY_VALUES or "không rõ" in value.lower() else value

    kind, title, org, year = get("loai"), get("ten"), get("don_vi"), get("nam")
    stem = filename.rsplit(".", 1)[0].lower()
    if not title or stem in title.lower():
        # SLM không chắc thường trả lại chính tên file ("[45.signed.pdf]") — vô ích, để người duyệt nhập
        return None
    if not re.fullmatch(r"(19|20)\d{2}", year):
        year = ""
    # Tiêu đề đã tự nói loại ("Luật Thương mại", "Mẫu hợp đồng ...") → không ghép thêm loại SLM đoán
    title_has_type = title.lower().startswith(_DOC_TYPE_WORDS)
    main = title if (not kind or title_has_type or kind.lower() in title.lower()) else f"{kind}: {title}"
    tail = ", ".join(x for x in (org, year) if x)
    try:
        header = normalize_header(f"[{_clip(main, 150)}{' — ' + _clip(tail, 70) if tail else ''}]")
    except ValueError:
        return None
    return HeaderSuggestion(header, SOURCE_SLM, "other", needs_review=True,
                            fields={"kind": kind, "title": title, "org": org, "year": year})


# ── Điều phối ───────────────────────────────────────────────────────────────

def ordered_texts(chunks: list[dict]) -> list[str]:
    """Nội dung chunk theo thứ tự đọc (trang, chunk_index) — chunk_index đánh lại theo trang."""
    def key(c):
        meta = c.get("metadata") or {}
        return (meta.get("page_num") or 0, meta.get("chunk_index") or 0)
    return [c["text"] for c in sorted(chunks, key=key)]


def suggest_header(filename: str, texts: list[str], *, slm_enabled: bool = False,
                   ollama_base_url: str = "http://localhost:11434", slm_model: str = "qwen2.5:3b-instruct",
                   slm_timeout: int = 60) -> HeaderSuggestion:
    """texts: nội dung chunk THEO THỨ TỰ ĐỌC (xem ordered_texts). Không bao giờ raise."""
    opening = repair_split_syllables(re.sub(r"\s+", " ", " ".join(texts[:OPENING_CHUNKS])))
    full_text = repair_split_syllables(re.sub(r"\s+", " ", " ".join(texts)))
    try:
        for rule in (
            lambda: _rule_excel(filename, opening),
            lambda: _rule_financial(filename, opening, full_text),
            lambda: _rule_legal(filename, opening, full_text),
        ):
            suggestion = rule()
            if suggestion:
                suggestion.header = normalize_header(suggestion.header)
                return suggestion
    except Exception as exc:  # noqa: BLE001 — gợi ý lỗi không được làm hỏng ingest
        logger.warning(f"Rule gợi ý header lỗi cho '{filename}' ({exc}) — thử SLM.")

    if slm_enabled and opening.strip():
        suggestion = _slm_suggest(filename, opening, ollama_base_url, slm_model, slm_timeout)
        if suggestion:
            return suggestion
    return HeaderSuggestion(None, None, "other", needs_review=True)
