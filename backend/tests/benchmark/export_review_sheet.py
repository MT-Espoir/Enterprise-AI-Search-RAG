"""
export_review_sheet.py — Xuất evaluation_dataset_full_candidates.json ra file Excel để
người duyệt (review_candidates_full.xlsx). Sau khi duyệt xong, chạy
apply_review_sheet.py để đưa kết quả về lại file JSON.

Cột người duyệt điền (nền vàng): Trạng thái (approved/rejected), Câu hỏi sửa lại,
Bỏ nhãn pooled, Ghi chú. Các cột còn lại chỉ để đọc — sửa ở đó KHÔNG có tác dụng.

Chạy: python backend/tests/benchmark/export_review_sheet.py [--force]
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BENCH_DIR, "benchmark_chromadb_full")
COLLECTION_NAME = "benchmark_full"
CANDIDATES_PATH = os.path.join(BENCH_DIR, "evaluation_dataset_full_candidates.json")
SHEET_PATH = os.path.join(BENCH_DIR, "review_candidates_full.xlsx")

REVIEW_SHEET = "Duyệt"
STATUSES = ("approved", "rejected", "pending")
SOURCE_PREVIEW_CHARS = 1500

# (header, width, key) — key dùng chung với apply_review_sheet.py
COLUMNS = [
    ("ID", 7, "id"),
    ("Trạng thái", 12, "status"),
    ("Câu hỏi sửa lại (tùy chọn)", 40, "question_edit"),
    ("Bỏ nhãn pooled (x)", 10, "drop_pooled"),
    ("Ghi chú", 25, "note"),
    ("Cờ cảnh báo", 20, "flags"),
    ("Nhóm", 14, "stratum"),
    ("Loại", 12, "category"),
    ("Câu hỏi", 50, "question"),
    ("Đáp án tham chiếu", 45, "reference_answer"),
    ("Trích dẫn (evidence)", 45, "evidence"),
    ("Tài liệu nguồn", 30, "source_file"),
    ("Tài liệu khác cũng đúng (pooled)", 30, "pooled_files"),
    ("Nội dung chunk nguồn", 80, "source_text"),
]
EDITABLE = {"status", "question_edit", "drop_pooled", "note"}

FONT = "Arial"
EDIT_FILL = PatternFill("solid", start_color="FFF2CC")
HEADER_FILL = PatternFill("solid", start_color="1F3864")
FLAG_FILL = PatternFill("solid", start_color="FCE4D6")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

FLAG_LABELS = {
    "source_not_confirmed": "Judge không xác nhận chunk nguồn (thường do judge bỏ sót — xem kỹ)",
    "answer_also_in_other_document": "Tài liệu khác cũng chứa đáp án — kiểm tra cột pooled",
}


def build_rows(candidates: list[dict]) -> list[dict]:
    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    source_ids = sorted({s["chunk_id"] for c in candidates for s in c["source_chunks"]})
    data = collection.get(ids=source_ids, include=["documents"])
    text_of = dict(zip(data["ids"], data["documents"]))

    rows = []
    for c in candidates:
        source_files = sorted({s["filename"] for s in c["source_chunks"]})
        pooled_files = sorted({r["filename"] for r in c["relevant_chunks"] if r["via"] == "pooled"})
        parts = [text_of[s["chunk_id"]] for s in c["source_chunks"]]
        budget = SOURCE_PREVIEW_CHARS // len(parts)
        source_text = "\n\n──── đoạn 2 ────\n".join(
            p[:budget] + ("…" if len(p) > budget else "") for p in parts
        )
        rows.append({
            "id": c["id"],
            "status": c["review"]["status"],
            "question_edit": "",
            "drop_pooled": "",
            "note": c["review"].get("note", ""),
            "flags": "\n".join(FLAG_LABELS.get(f, f) for f in c["flags"]),
            "stratum": c["stratum"],
            "category": c["category"],
            "question": c["question"],
            "reference_answer": c.get("reference_answer") or "",
            "evidence": c.get("evidence") or "",
            "source_file": "\n".join(source_files),
            "pooled_files": "\n".join(pooled_files),
            "source_text": source_text,
        })
    return rows


def write_guide(wb: Workbook):
    ws = wb.create_sheet("Hướng dẫn", 0)
    lines = [
        ("Cách duyệt benchmark — sheet \"Duyệt\"", True),
        ("", False),
        ("Chỉ điền ở 4 cột NỀN VÀNG. Các cột khác chỉ để đọc, sửa ở đó không có tác dụng.", False),
        ("", False),
        ("Trạng thái: approved = đưa vào benchmark · rejected = loại · pending = chưa duyệt (sẽ bị BỎ QUA khi chốt).", False),
        ("Câu hỏi sửa lại: để trống nếu giữ nguyên. Điền câu mới nếu muốn sửa câu hỏi (vd làm rõ tên văn bản/năm).", False),
        ("Bỏ nhãn pooled: điền x nếu các tài liệu ở cột \"Tài liệu khác cũng đúng\" thực ra KHÔNG trả lời được câu hỏi.", False),
        ("Ghi chú: tùy ý, được lưu lại cùng câu hỏi.", False),
        ("", False),
        ("Tiêu chí approved:", True),
        ("  1. Câu hỏi tự đủ nghĩa (nêu rõ văn bản/công ty/năm) — không phải \"văn bản này\".", False),
        ("  2. Trả lời được từ chunk nguồn (cột cuối) — đáp án tham chiếu khớp nội dung chunk.", False),
        ("  3. Giống câu một nhân viên thật sẽ hỏi; không hỏi về chữ ký, số trang, mục lục.", False),
        ("  4. Không mơ hồ về thời điểm (vd \"giai đoạn gần nhất\" → nên sửa thành năm cụ thể hoặc reject).", False),
        ("", False),
        ("Ví dụ một dòng đã duyệt:", True),
        ("  Trạng thái = approved · Câu hỏi sửa lại = (trống) · Bỏ nhãn pooled = (trống) · Ghi chú = (trống)", False),
        ("  Trạng thái = approved · Câu hỏi sửa lại = \"Tỷ lệ tăng trưởng EPS năm 2024 của Vinamilk là bao nhiêu?\"", False),
        ("  Trạng thái = rejected · Ghi chú = \"chunk nguồn lỗi font, câu hỏi chung chung\"", False),
        ("", False),
        ("Cờ cảnh báo (cột F, nền cam):", True),
    ] + [(f"  • {label}", False) for label in FLAG_LABELS.values()] + [
        ("", False),
        ("Duyệt xong: lưu file, rồi chạy", True),
        ("  python backend/tests/benchmark/apply_review_sheet.py", False),
        ("  python backend/tests/benchmark/finalize_benchmark_dataset.py", False),
    ]
    for i, (text, bold) in enumerate(lines, 1):
        cell = ws.cell(row=i, column=1, value=text)
        cell.font = Font(name=FONT, bold=bold, size=12 if i == 1 else 10)
    ws.column_dimensions["A"].width = 120


def write_review(wb: Workbook, rows: list[dict]):
    ws = wb.create_sheet(REVIEW_SHEET)
    for col, (header, width, _) in enumerate(COLUMNS, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BORDER
        ws.column_dimensions[cell.column_letter].width = width

    for r, row in enumerate(rows, 2):
        for col, (_, _, key) in enumerate(COLUMNS, 1):
            cell = ws.cell(row=r, column=col, value=row[key])
            cell.font = Font(name=FONT, size=10)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = BORDER
            if key in EDITABLE:
                cell.fill = EDIT_FILL
            elif key == "flags" and row["flags"]:
                cell.fill = FLAG_FILL
        ws.row_dimensions[r].height = 150

    status_col = next(i for i, c in enumerate(COLUMNS, 1) if c[2] == "status")
    letter = ws.cell(row=1, column=status_col).column_letter
    dv = DataValidation(type="list", formula1='"' + ",".join(STATUSES) + '"', allow_blank=False,
                        showErrorMessage=True, errorTitle="Trạng thái", error="Chọn approved / rejected / pending")
    ws.add_data_validation(dv)
    dv.add(f"{letter}2:{letter}{len(rows) + 1}")

    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(COLUMNS)).column_letter}{len(rows) + 1}"


def write_summary(wb: Workbook, rows: list[dict]):
    ws = wb.create_sheet("Tổng hợp")
    status_letter = "B"
    stratum_letter = "G"
    n = len(rows) + 1
    rng = lambda col: f"'{REVIEW_SHEET}'!${col}$2:${col}${n}"

    ws["A1"], ws["B1"] = "Trạng thái", "Số câu"
    for i, status in enumerate(STATUSES, 2):
        ws[f"A{i}"] = status
        ws[f"B{i}"] = f'=COUNTIF({rng(status_letter)},A{i})'
    ws["A5"], ws["B5"] = "Tổng", "=SUM(B2:B4)"

    ws["A7"], ws["B7"], ws["C7"], ws["D7"] = "Nhóm", "Tổng", "approved", "rejected"
    for i, stratum in enumerate(sorted({r["stratum"] for r in rows}), 8):
        ws[f"A{i}"] = stratum
        ws[f"B{i}"] = f'=COUNTIF({rng(stratum_letter)},A{i})'
        ws[f"C{i}"] = f'=COUNTIFS({rng(stratum_letter)},A{i},{rng(status_letter)},"approved")'
        ws[f"D{i}"] = f'=COUNTIFS({rng(stratum_letter)},A{i},{rng(status_letter)},"rejected")'

    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name=FONT, bold=cell.row in (1, 5, 7))
    ws.column_dimensions["A"].width = 20


def run(force: bool):
    if os.path.exists(SHEET_PATH) and not force:
        sys.exit(f"❌ {SHEET_PATH} đã tồn tại — có thể chứa kết quả duyệt dở. Dùng --force để ghi đè.")
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        candidates = json.load(f)["candidates"]

    rows = build_rows(candidates)
    wb = Workbook()
    wb.remove(wb.active)
    write_guide(wb)
    write_review(wb, rows)
    write_summary(wb, rows)
    wb.active = 1
    # openpyxl không lưu giá trị đã tính của công thức → buộc Excel tính lại khi mở file
    wb.calculation.fullCalcOnLoad = True
    wb.save(SHEET_PATH)
    print(f"✅ {len(rows)} câu → {SHEET_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ghi đè file Excel đã có")
    run(force=parser.parse_args().force)
