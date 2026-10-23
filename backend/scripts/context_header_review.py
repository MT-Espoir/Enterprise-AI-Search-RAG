"""
context_header_review.py — S4 (contextual_header_rollout_plan.md): gợi ý header ngữ cảnh HÀNG LOẠT cho
tài liệu đã có, để người duyệt qua Excel, rồi mới ghi vào MongoDB.

  suggest  ĐỌC MongoDB + ChromaDB production, chạy rule → SLM local cho từng tài liệu → output/suggestions.json.
           KHÔNG ghi gì vào production. Bổ sung:
             - 21 header BCTC user đã duyệt ở PoC (tests/benchmark/poc_contextual_headers.json) → điền sẵn "approved"
             - tài liệu TRÙNG NỘI DUNG (chunk giống hệt, hoặc MANUAL_EQUIVALENT_DOCS đã xác minh tay): header rule/PoC
               của bản này được đề xuất cho bản kia (vd 59.signed.pdf ≡ luatdoanhnghiep.pdf)
  export   suggestions.json → output/context_header_review.xlsx (sheet "Duyệt": 3 cột nền vàng người duyệt điền)
  apply    đọc Excel đã duyệt → ghi MongoDB. MẶC ĐỊNH CHẠY THỬ (chỉ in thay đổi); thêm --commit mới ghi thật.
           Chỉ đổi trường header trong MongoDB — KHÔNG embed lại (đó là S6), killswitch tắt thì hành vi tìm kiếm
           không đổi.

Chạy (từ thư mục gốc repo):
  ml_env/Scripts/python.exe backend/scripts/context_header_review.py suggest [--no-slm]
  ml_env/Scripts/python.exe backend/scripts/context_header_review.py export
  ml_env/Scripts/python.exe backend/scripts/context_header_review.py apply            # chạy thử
  ml_env/Scripts/python.exe backend/scripts/context_header_review.py apply --commit   # ghi thật
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(BACKEND, ".env"))

from app.core.context_header_suggest import (  # noqa: E402
    normalize_header, ordered_texts, repair_split_syllables, suggest_header,
)
from app.services.context_header_service import plan_header_update  # noqa: E402

OUTPUT_DIR = os.path.join(BACKEND, "scripts", "output")
SUGGESTIONS_PATH = os.path.join(OUTPUT_DIR, "context_header_suggestions.json")
SHEET_PATH = os.path.join(OUTPUT_DIR, "context_header_review.xlsx")
POC_HEADERS_PATH = os.path.join(BACKEND, "tests", "benchmark", "poc_contextual_headers.json")
REVIEW_SHEET = "Duyệt"
STATUSES = ("approved", "no_header", "pending")
OPENING_PREVIEW_CHARS = 1200
SOURCE_POC = "poc"
SOURCE_TWIN = "twin"


# ── Đọc production (CHỈ ĐỌC) ────────────────────────────────────────────────

def _mongo_db():
    from pymongo import MongoClient
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/rag_db")
    return MongoClient(uri).get_default_database()


def _chroma_collection():
    import chromadb
    path = (os.getenv("CHROMA_PERSIST_PATH") or "").strip() or os.path.abspath(os.path.join(BACKEND, "..", "chromadb_data"))
    return chromadb.PersistentClient(path=path).get_collection(os.getenv("CHROMA_COLLECTION_NAME", "enterprise_documents"))


def _norm_hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def equivalence_groups(chunk_hashes: dict[str, set[str]], manual: list[list[str]], overlap: float = 0.5) -> list[list[str]]:
    """Nhóm tài liệu trùng nội dung: ≥ overlap chunk giống hệt (của tài liệu nhỏ hơn) + danh sách xác minh tay."""
    parent = {name: name for name in chunk_hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    names = sorted(chunk_hashes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            smaller = min(len(chunk_hashes[a]), len(chunk_hashes[b]))
            if smaller and len(chunk_hashes[a] & chunk_hashes[b]) / smaller >= overlap:
                parent[find(a)] = find(b)
    for group in manual:
        present = [n for n in group if n in parent]
        for other in present[1:]:
            parent[find(other)] = find(present[0])
    groups = defaultdict(list)
    for name in names:
        groups[find(name)].append(name)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def propagate_twins(rows: list[dict], groups: list[list[str]]) -> None:
    """Trong 1 nhóm trùng nội dung, tài liệu không có header tin cậy nhận header tin cậy (PoC / rule không cần
    xem lại) của bản trùng — vẫn để người duyệt xác nhận."""
    by_name = {r["filename"]: r for r in rows}
    for group in groups:
        members = [by_name[n] for n in group if n in by_name]
        for row in members:
            row["equivalent_to"] = [n for n in group if n != row["filename"]]
        trusted = [r for r in members if r["source"] == SOURCE_POC or (r["source"] == "rule" and not r["needs_review"])]
        if not trusted:
            continue
        donor = trusted[0]
        for row in members:
            if row in trusted:
                continue
            row["system_header"] = row.get("system_header") or row["header"]
            row.update(header=donor["header"], source=SOURCE_TWIN, needs_review=True,
                       note=f"Lấy từ bản trùng nội dung: {donor['filename']}")


def cmd_suggest(slm_enabled: bool):
    from app.services.context_header_service import header_state  # noqa: F401 — kiểm tra import sớm
    from tests.benchmark.finalize_benchmark_dataset import MANUAL_EQUIVALENT_DOCS

    db, col = _mongo_db(), _chroma_collection()
    docs = list(db.documents.find({"status": "done"}, {"chroma_ids": 0}).sort("original_name", 1))
    poc = {}
    if os.path.exists(POC_HEADERS_PATH):
        with open(POC_HEADERS_PATH, encoding="utf-8") as f:
            poc = {h["filename"]: h["header"] for h in json.load(f)["headers"]}
    print(f"📋 {len(docs)} tài liệu (status=done) — DB {db.name}; header PoC đã duyệt: {len(poc)}; SLM {'BẬT' if slm_enabled else 'TẮT'}\n")

    rows, chunk_hashes = [], {}
    for i, doc in enumerate(docs, 1):
        doc_id, name = str(doc["_id"]), doc.get("original_name") or doc.get("filename")
        raw = col.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        chunks = [{"text": t, "metadata": m} for t, m in zip(raw["documents"], raw["metadatas"])]
        texts = ordered_texts(chunks)
        chunk_hashes[name] = {_norm_hash(t) for t in texts}
        if name in poc:
            header, source, kind, needs_review = normalize_header(poc[name]), SOURCE_POC, "financial_report", False
        else:
            s = suggest_header(name, texts, slm_enabled=slm_enabled,
                               ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                               slm_model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b-instruct"))
            header, source, kind, needs_review = s.header, s.source, s.kind, s.needs_review
        opening = repair_split_syllables(re.sub(r"\s+", " ", " ".join(texts[:4])))[:OPENING_PREVIEW_CHARS]
        rows.append({
            "doc_id": doc_id, "filename": name, "chunk_count": len(texts), "kind": kind, "source": source,
            "needs_review": needs_review, "header": header, "system_header": None, "note": "",
            "equivalent_to": [], "opening": opening,
            "current_approved": doc.get("context_header"), "current_suggested": doc.get("context_header_suggested"),
        })
        print(f"[{i}/{len(docs)}] {'?' if needs_review else ' '} {(source or '-'):<4} {name[:45]:<45} {header}")

    propagate_twins(rows, equivalence_groups(chunk_hashes, MANUAL_EQUIVALENT_DOCS))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUGGESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump({"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "db": db.name, "slm_enabled": slm_enabled, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"\n💾 {SUGGESTIONS_PATH}")


# ── Excel ───────────────────────────────────────────────────────────────────

# (header, width, key) — key dùng chung cho export và apply
COLUMNS = [
    ("Doc ID", 26, "doc_id"),
    ("Tên file", 34, "filename"),
    ("Trạng thái", 13, "status"),
    ("Header chốt", 60, "final_header"),
    ("Ghi chú", 24, "review_note"),
    ("Nguồn gợi ý", 12, "source"),
    ("Cần xem kỹ", 10, "needs_review"),
    ("Header hệ thống gợi ý", 50, "header"),
    ("Thông tin thêm", 30, "info"),
    ("Phần mở đầu tài liệu", 90, "opening"),
]
EDITABLE = {"status", "final_header", "review_note"}
SOURCE_LABELS = {
    SOURCE_POC: "PoC (đã duyệt)", "rule": "rule", "slm": "SLM", SOURCE_TWIN: "bản trùng", None: "—",
}


def cmd_export(force: bool):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    if os.path.exists(SHEET_PATH) and not force:
        sys.exit(f"❌ {SHEET_PATH} đã tồn tại — có thể chứa kết quả duyệt dở. Dùng --force để ghi đè.")
    with open(SUGGESTIONS_PATH, encoding="utf-8") as f:
        rows = json.load(f)["rows"]

    font = "Arial"
    edit_fill, header_fill, flag_fill = (PatternFill("solid", start_color=c) for c in ("FFF2CC", "1F3864", "FCE4D6"))
    side = Side(style="thin", color="BFBFBF")
    border = Border(left=side, right=side, top=side, bottom=side)
    wb = Workbook()
    wb.remove(wb.active)

    guide = wb.create_sheet("Hướng dẫn")
    lines = [
        ("Duyệt header ngữ cảnh cho tài liệu — sheet \"Duyệt\"", True),
        ("", False),
        ("Header ngữ cảnh = 1 dòng NHẬN DIỆN tài liệu (loại, số hiệu/năm, đơn vị) được ghép vào trước mọi đoạn khi tìm kiếm,", False),
        ("giúp phân biệt các tài liệu rất giống nhau (vd báo cáo tài chính cùng công ty qua nhiều năm). Không tóm tắt nội dung.", False),
        ("", False),
        ("Chỉ điền 3 cột NỀN VÀNG:", True),
        ("  Trạng thái: approved = dùng \"Header chốt\" · no_header = tài liệu này KHÔNG dùng header · pending = chưa quyết (bỏ qua, không ghi)", False),
        ("  Header chốt: đã điền sẵn bằng gợi ý của hệ thống — sửa trực tiếp nếu chưa đúng. Tối đa 250 ký tự, 1 dòng.", False),
        ("  Ghi chú: tùy ý.", False),
        ("", False),
        ("Lưu ý:", True),
        ("  • Nên để MỌI tài liệu có header: tài liệu không có header bị thiệt khi tìm kiếm so với tài liệu có (đã đo ở PoC).", False),
        ("  • Dòng nền cam = \"Cần xem kỹ\": gợi ý từ SLM, từ bản trùng, hoặc rule thiếu thông tin (vd không rõ riêng/hợp nhất).", False),
        ("  • 21 báo cáo tài chính đã được bạn duyệt ở PoC → điền sẵn approved.", False),
        ("  • Cột \"Phần mở đầu tài liệu\" để đối chiếu; bản ký số thường mất trang tiêu đề — cần tự nhập.", False),
        ("", False),
        ("Mẫu tham khảo:", True),
        ("  [Báo cáo tài chính riêng năm 2016 của Tổng Công ty Cổ phần Xuất nhập khẩu và Xây dựng Việt Nam (Vinaconex), năm tài chính kết thúc ngày 31/12/2016]", False),
        ("  [Nghị định số 145/2020/NĐ-CP quy định chi tiết một số điều của Bộ luật Lao động về điều kiện lao động và quan hệ lao động]", False),
        ("  [Lịch sử giá của Công ty Cổ phần Đầu tư Thế Giới Di Động (mã CK: MWG) — nguồn Simplize]", False),
        ("", False),
        ("Duyệt xong: lưu file, báo lại để chạy bước apply (chạy thử trước, ghi thật sau khi bạn đồng ý).", True),
    ]
    for i, (text, bold) in enumerate(lines, 1):
        guide.cell(row=i, column=1, value=text).font = Font(name=font, bold=bold, size=12 if i == 1 else 10)
    guide.column_dimensions["A"].width = 150

    ws = wb.create_sheet(REVIEW_SHEET)
    for c, (title, width, _) in enumerate(COLUMNS, 1):
        cell = ws.cell(row=1, column=c, value=title)
        cell.font, cell.fill, cell.border = Font(name=font, bold=True, color="FFFFFF"), header_fill, border
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[cell.column_letter].width = width
    for r, row in enumerate(rows, 2):
        info = []
        if row["equivalent_to"]:
            info.append("Trùng nội dung với: " + ", ".join(row["equivalent_to"]))
        if row.get("system_header") and row["system_header"] != row["header"]:
            info.append(f"Gợi ý gốc của chính tài liệu: {row['system_header']}")
        if row["note"]:
            info.append(row["note"])
        values = {
            **row,
            "status": "approved" if row["source"] == SOURCE_POC else "pending",
            "final_header": row["header"] or "",
            "review_note": "",
            "source": SOURCE_LABELS.get(row["source"], row["source"]),
            "needs_review": "x" if row["needs_review"] else "",
            "header": row["header"] or "(không gợi ý được)",
            "info": "\n".join(info),
        }
        for c, (_, _, key) in enumerate(COLUMNS, 1):
            cell = ws.cell(row=r, column=c, value=values[key])
            cell.font, cell.border = Font(name=font, size=10), border
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if key in EDITABLE:
                cell.fill = edit_fill
            elif row["needs_review"] or not row["header"]:
                cell.fill = flag_fill
        ws.row_dimensions[r].height = 90

    status_letter = ws.cell(row=1, column=[k for _, _, k in COLUMNS].index("status") + 1).column_letter
    dv = DataValidation(type="list", formula1='"' + ",".join(STATUSES) + '"', allow_blank=False,
                        showErrorMessage=True, errorTitle="Trạng thái", error="Chọn approved / no_header / pending")
    ws.add_data_validation(dv)
    dv.add(f"{status_letter}2:{status_letter}{len(rows) + 1}")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(COLUMNS)).column_letter}{len(rows) + 1}"

    summary = wb.create_sheet("Tổng hợp")
    rng = f"'{REVIEW_SHEET}'!${status_letter}$2:${status_letter}${len(rows) + 1}"
    summary["A1"], summary["B1"] = "Trạng thái", "Số tài liệu"
    for i, status in enumerate(STATUSES, 2):
        summary[f"A{i}"], summary[f"B{i}"] = status, f"=COUNTIF({rng},A{i})"
    summary["A5"], summary["B5"] = "Tổng", "=SUM(B2:B4)"
    for row_cells in summary.iter_rows():
        for cell in row_cells:
            cell.font = Font(name=font, bold=cell.row in (1, 5))
    summary.column_dimensions["A"].width = 16

    wb.active = 1
    wb.calculation.fullCalcOnLoad = True  # openpyxl không lưu giá trị công thức → Excel tự tính khi mở
    wb.save(SHEET_PATH)
    print(f"✅ {len(rows)} tài liệu → {SHEET_PATH}")


# ── Apply ───────────────────────────────────────────────────────────────────

def read_review_sheet(path: str) -> dict[str, dict]:
    from openpyxl import load_workbook

    ws = load_workbook(path, read_only=True, data_only=True)[REVIEW_SHEET]
    rows = ws.iter_rows(values_only=True)
    titles = [str(c or "").strip() for c in next(rows)]
    if titles[:len(COLUMNS)] != [t for t, _, _ in COLUMNS]:
        sys.exit("❌ Tiêu đề cột đã bị đổi — không xác định được cột người duyệt điền.")
    keys = [k for _, _, k in COLUMNS]
    out = {}
    for values in rows:
        row = dict(zip(keys, values))
        doc_id = str(row["doc_id"] or "").strip()
        if not doc_id:
            continue
        if doc_id in out:
            sys.exit(f"❌ Doc ID {doc_id} xuất hiện 2 lần.")
        out[doc_id] = {
            "status": str(row["status"] or "pending").strip().lower(),
            "final_header": str(row["final_header"] or "").strip(),
            "review_note": str(row["review_note"] or "").strip(),
        }
    return out


def plan_review_changes(rows: list[dict], reviews: dict[str, dict], current_docs: dict[str, dict]) -> list[dict]:
    """Thay đổi cần ghi cho từng tài liệu — hàm THUẦN (test không cần Mongo). Lỗi dữ liệu → ValueError."""
    known = {r["doc_id"]: r for r in rows}
    unknown = sorted(set(reviews) - set(known))
    if unknown:
        raise ValueError(f"Doc ID không có trong danh sách gợi ý: {unknown[:5]}")
    bad = {d: r["status"] for d, r in reviews.items() if r["status"] not in STATUSES}
    if bad:
        raise ValueError(f"Trạng thái không hợp lệ: {bad}")

    changes = []
    for doc_id, review in reviews.items():
        if review["status"] == "pending":
            continue
        row, doc = known[doc_id], current_docs.get(doc_id)
        if doc is None:
            raise ValueError(f"Tài liệu {doc_id} ({row['filename']}) không còn trong MongoDB.")
        if review["status"] == "approved":
            approved = normalize_header(review["final_header"])
            if not approved:
                raise ValueError(f"{row['filename']}: trạng thái approved nhưng 'Header chốt' rỗng.")
            update = plan_header_update(doc, approved=approved, suggested=row["header"],
                                        suggested_source=row["source"], needs_review=row["needs_review"])
        else:  # no_header
            update = plan_header_update(doc, approved=None, suggested=None, suggested_source=None, needs_review=False)
        if review["review_note"]:
            update["context_header_review_note"] = review["review_note"]
        changes.append({"doc_id": doc_id, "filename": row["filename"], "status": review["status"], "update": update,
                        "before": doc.get("context_header") or doc.get("context_header_suggested"),
                        "after": update["context_header"] or update["context_header_suggested"]})
    return changes


def cmd_apply(commit: bool):
    from bson import ObjectId

    with open(SUGGESTIONS_PATH, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    reviews = read_review_sheet(SHEET_PATH)
    db = _mongo_db()
    current = {str(d["_id"]): d for d in db.documents.find({"_id": {"$in": [ObjectId(i) for i in reviews]}})}
    try:
        changes = plan_review_changes(rows, reviews, current)
    except ValueError as exc:
        sys.exit(f"❌ {exc}")

    counts = defaultdict(int)
    for ch in changes:
        counts[ch["status"]] += 1
        marker = "=" if ch["before"] == ch["after"] else "→"
        print(f"  {ch['status']:<9} {ch['filename'][:40]:<40} {marker} {ch['after']}")
    pending = sum(1 for r in reviews.values() if r["status"] == "pending")
    print(f"\n{dict(counts)} · pending (bỏ qua) {pending} · DB {db.name}")
    if not commit:
        print("ℹ️  CHẠY THỬ — chưa ghi gì. Thêm --commit để ghi vào MongoDB.")
        return
    now = datetime.now(timezone.utc)
    for ch in changes:
        db.documents.update_one({"_id": ObjectId(ch["doc_id"])}, {"$set": {**ch["update"], "updated_at": now}})
    print(f"✅ Đã ghi {len(changes)} tài liệu vào {db.name}. Embedding chưa đổi — cần embed lại (S6).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("suggest")
    p.add_argument("--no-slm", action="store_true", help="Chỉ dùng rule, không gọi SLM")
    p = sub.add_parser("export")
    p.add_argument("--force", action="store_true", help="Ghi đè file Excel đã có")
    p = sub.add_parser("apply")
    p.add_argument("--commit", action="store_true", help="Ghi thật vào MongoDB (mặc định chạy thử)")
    args = parser.parse_args()
    if args.cmd == "suggest":
        cmd_suggest(slm_enabled=not args.no_slm)
    elif args.cmd == "export":
        cmd_export(force=args.force)
    else:
        cmd_apply(commit=args.commit)
