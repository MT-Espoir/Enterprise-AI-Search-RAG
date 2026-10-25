"""
export_review_sheet.py — Xuất evaluation_dataset_full_candidates.json ra file Excel để
người duyệt (review_candidates_full.xlsx). Sau khi duyệt xong, chạy
apply_review_sheet.py để đưa kết quả về lại file JSON.

Cột người duyệt điền (nền vàng): Trạng thái (approved/rejected), Câu hỏi sửa lại,
Bỏ nhãn pooled, Ghi chú. Các cột còn lại chỉ để đọc — sửa ở đó KHÔNG có tác dụng.

--set vague: bộ câu hỏi "mơ hồ" (generate_vague_questions.py) — file riêng, tiêu chí duyệt riêng.

Chạy: python backend/tests/benchmark/export_review_sheet.py [--force] [--set full|vague]
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
# Bộ câu hỏi: full = 131 câu tự đủ nghĩa (nêu rõ văn bản/năm); vague = câu ngắn KHÔNG nêu năm/loại/
# số hiệu (contextual_header_rollout_plan.md §4.4) — đo header có làm câu mơ hồ tệ đi không
SETS = {
    "full": {"candidates": CANDIDATES_PATH, "sheet": SHEET_PATH,
             "output": os.path.join(BENCH_DIR, "evaluation_dataset_full.json")},
    "vague": {"candidates": os.path.join(BENCH_DIR, "evaluation_dataset_vague_candidates.json"),
              "sheet": os.path.join(BENCH_DIR, "review_candidates_vague.xlsx"),
              "output": os.path.join(BENCH_DIR, "evaluation_dataset_vague.json")},
    # 70 câu chia 4 nhóm (generate_ambiguous_questions.py) — mở rộng bộ "vague" 30 câu toàn câu ngắn
    "vague70": {"candidates": os.path.join(BENCH_DIR, "evaluation_dataset_vague70_candidates.json"),
                "sheet": os.path.join(BENCH_DIR, "review_candidates_vague70.xlsx"),
                "output": os.path.join(BENCH_DIR, "evaluation_dataset_vague70.json")},
}
# Số câu mục tiêu mỗi nhóm câu hỏi (người duyệt chốt đúng số này) + nhãn hiển thị trong Excel
SET_TARGETS = {"vague70": {"broad_short": 25, "paraphrase": 20, "underspecified": 10, "multi_doc": 15}}
CATEGORY_LABELS = {
    "broad_short": "1. Rộng, ngắn, không dùng thuật ngữ",
    "paraphrase": "2. Gián tiếp / đồng nghĩa",
    "underspecified": "3. Thiếu ngữ cảnh, xác định được đối tượng",
    "multi_doc": "4. Tổng hợp nhiều tài liệu",
}

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
# Bộ mơ hồ: câu không nêu năm → BCTC các năm khác cũng trả lời được nhưng pooling (top-10 mỗi hệ
# thống) có thể không thấy hết → người duyệt bổ sung tên file, tính đúng ở MỨC TÀI LIỆU khi chấm
_EXTRA_FILES_COLUMN = ("Thêm tài liệu cũng đúng (tên file, mỗi dòng 1 file)", 38, "extra_files")
EXTRA_COLUMNS = {"vague": [_EXTRA_FILES_COLUMN], "vague70": [_EXTRA_FILES_COLUMN]}
EDITABLE = {"status", "question_edit", "drop_pooled", "note", "extra_files"}


def columns_for(kind: str) -> list[tuple]:
    return COLUMNS + EXTRA_COLUMNS.get(kind, [])

FONT = "Arial"
EDIT_FILL = PatternFill("solid", start_color="FFF2CC")
HEADER_FILL = PatternFill("solid", start_color="1F3864")
FLAG_FILL = PatternFill("solid", start_color="FCE4D6")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

FLAG_LABELS = {
    "source_not_confirmed": "Judge không xác nhận chunk nguồn (thường do judge bỏ sót — xem kỹ)",
    "answer_also_in_other_document": "Tài liệu khác cũng chứa đáp án — kiểm tra cột pooled",
    "mentions_year_type_or_number": "Câu hỏi lỡ nêu năm / loại báo cáo / số hiệu — bộ mơ hồ: sửa bỏ hoặc reject",
    "source_text_garbled": "Chunk nguồn khó đọc (OCR lỗi / chữ đảo ngược) — kiểm tra đáp án CÓ THẬT trong chunk, LLM có thể bịa",
    "evidence_not_in_source": "Trích dẫn của LLM không khớp chunk nguồn — kiểm tra đáp án có căn cứ không",
}

CRITERIA = {
    "full": [
        "  1. Câu hỏi tự đủ nghĩa (nêu rõ văn bản/công ty/năm) — không phải \"văn bản này\".",
        "  2. Trả lời được từ chunk nguồn (cột cuối) — đáp án tham chiếu khớp nội dung chunk.",
        "  3. Giống câu một nhân viên thật sẽ hỏi; không hỏi về chữ ký, số trang, mục lục.",
        "  4. Không mơ hồ về thời điểm (vd \"giai đoạn gần nhất\" → nên sửa thành năm cụ thể hoặc reject).",
    ],
    "vague": [
        "  Bộ này CỐ Ý mơ hồ: câu ngắn kiểu người dùng gõ vội, KHÔNG nêu năm / loại báo cáo (riêng, hợp nhất) /",
        "  số hiệu / tên đầy đủ văn bản. Đừng sửa câu cho rõ ràng hơn — đó chính là thứ cần đo.",
        "  1. Câu NGẮN, tự nhiên, không nêu năm/loại/số hiệu. Câu nào lỡ nêu → sửa bỏ đi hoặc reject.",
        "  2. Trả lời được từ chunk nguồn (cột cuối) — đáp án tham chiếu khớp nội dung chunk.",
        "  3. Cột \"Tài liệu khác cũng đúng\": tài liệu mà câu hỏi NHƯ ĐÃ ĐẶT cũng trả lời được (vd cùng chỉ tiêu",
        "     ở BCTC năm khác) — khi chấm, hệ thống lấy được BẤT KỲ tài liệu nào trong đó đều tính ĐÚNG.",
        "     Nếu các tài liệu đó thực ra không trả lời được → điền x ở cột \"Bỏ nhãn pooled\".",
        "  4. Reject nếu mơ hồ tới mức không ai trả lời được (vd \"quy định là gì?\"), hoặc hỏi chữ ký/số trang/mục lục.",
        "  5. Cột cuối \"Thêm tài liệu cũng đúng\": nếu biết tài liệu khác cũng trả lời được câu hỏi như đã đặt mà",
        "     chưa có ở cột pooled (vd BCTC Vinaconex các năm khác cho câu không nêu năm) → ghi TÊN FILE, mỗi dòng",
        "     1 file (đúng tên trong trang Tài liệu). Chỉ tính ở mức tài liệu; tên sai sẽ bị báo lỗi khi chốt.",
    ],
    "vague70": [
        "  Bộ 70 câu kiểu người dùng thật, chia 4 NHÓM (cột \"Loại\"). Mỗi nhóm sinh DƯ ~3 câu → chỉ approved ĐÚNG",
        "  số mục tiêu: nhóm 1 = 25 · nhóm 2 = 20 · nhóm 3 = 10 · nhóm 4 = 15 (sheet \"Tổng hợp\" đếm sẵn). Đừng sửa câu",
        "  cho rõ ràng hơn — độ mơ hồ là thứ cần đo.",
        "  Nhóm 1 · Rộng, ngắn, KHÔNG dùng thuật ngữ trong tài liệu: câu ngắn kiểu gõ vội, dùng từ đời thường thay thuật ngữ.",
        "  Nhóm 2 · Gián tiếp / đồng nghĩa: mô tả tình huống hoặc thay thuật ngữ bằng từ đồng nghĩa — không lặp từ khóa.",
        "  Nhóm 3 · Thiếu ngữ cảnh nhưng XÁC ĐỊNH ĐƯỢC ĐỐI TƯỢNG: không nêu tên công ty/văn bản mà mô tả gián tiếp;",
        "     cột \"Trích dẫn\" ghi [Đối tượng: ...] — reject nếu mô tả KHÔNG chỉ tới đúng 1 đối tượng đó.",
        "  Nhóm 4 · Tổng hợp nhiều tài liệu: cần thông tin từ CẢ 2 tài liệu nguồn (cột \"Tài liệu nguồn\") — reject nếu",
        "     chỉ đọc 1 tài liệu đã trả lời đủ. Khi chấm, ngoài hit@k còn đo hệ thống có lấy ĐỦ cả 2 tài liệu không.",
        "  Chung: trả lời được từ chunk nguồn; nhóm 1–3 không nêu năm / loại báo cáo / số hiệu (cờ cam nếu lỡ nêu);",
        "  cột \"Tài liệu khác cũng đúng\" + cột cuối \"Thêm tài liệu cũng đúng\" như bộ 30 câu (tài liệu bổ sung tính",
        "  đúng ở mức tài liệu; tên file sai bị báo lỗi khi chốt). Reject câu hỏi chữ ký / số trang / mục lục.",
    ],
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
            "category": CATEGORY_LABELS.get(c["category"], c["category"]),
            "question": c["question"],
            "reference_answer": c.get("reference_answer") or "",
            "evidence": c.get("evidence") or "",
            "source_file": "\n".join(source_files),
            "pooled_files": "\n".join(pooled_files),
            "source_text": source_text,
            "extra_files": "\n".join(c.get("extra_relevant_files", [])),
        })
    return rows


def write_guide(wb: Workbook, kind: str = "full"):
    ws = wb.create_sheet("Hướng dẫn", 0)
    lines = [
        ("Cách duyệt benchmark — sheet \"Duyệt\"", True),
        ("", False),
        ("Chỉ điền ở 4 cột NỀN VÀNG. Các cột khác chỉ để đọc, sửa ở đó không có tác dụng.", False),
        ("", False),
        ("Trạng thái: approved = đưa vào benchmark · rejected = loại · pending = chưa duyệt (sẽ bị BỎ QUA khi chốt).", False),
        ("Câu hỏi sửa lại: để trống nếu giữ nguyên. Điền câu mới nếu muốn sửa câu hỏi "
         + ("(vd làm rõ tên văn bản/năm)." if kind == "full" else "(vd bỏ năm/số hiệu lỡ nêu — KHÔNG thêm cho rõ hơn)."), False),
        ("Bỏ nhãn pooled: điền x nếu các tài liệu ở cột \"Tài liệu khác cũng đúng\" thực ra KHÔNG trả lời được câu hỏi.", False),
        ("Ghi chú: tùy ý, được lưu lại cùng câu hỏi.", False),
        ("", False),
        ("Tiêu chí approved:", True),
    ] + [(line, False) for line in CRITERIA[kind]] + [
        ("", False),
        ("Ví dụ một dòng đã duyệt:", True),
        ("  Trạng thái = approved · Câu hỏi sửa lại = (trống) · Bỏ nhãn pooled = (trống) · Ghi chú = (trống)", False),
        (("  Trạng thái = approved · Câu hỏi sửa lại = \"Tỷ lệ tăng trưởng EPS năm 2024 của Vinamilk là bao nhiêu?\""
          if kind == "full" else
          "  Trạng thái = approved · Câu hỏi sửa lại = \"Vinaconex chia cổ tức bao nhiêu?\" (bỏ năm khỏi câu gốc)"), False),
        ("  Trạng thái = rejected · Ghi chú = \"chunk nguồn lỗi font, câu hỏi chung chung\"", False),
        ("", False),
        ("Cờ cảnh báo (cột F, nền cam):", True),
    ] + [(f"  • {label}", False) for label in FLAG_LABELS.values()] + [
        ("", False),
        ("Duyệt xong: lưu file, rồi chạy", True),
        (f"  python backend/tests/benchmark/apply_review_sheet.py --set {kind}", False),
        (f"  python backend/tests/benchmark/finalize_benchmark_dataset.py --set {kind}", False),
    ]
    for i, (text, bold) in enumerate(lines, 1):
        cell = ws.cell(row=i, column=1, value=text)
        cell.font = Font(name=FONT, bold=bold, size=12 if i == 1 else 10)
    ws.column_dimensions["A"].width = 120


def write_review(wb: Workbook, rows: list[dict], kind: str = "full"):
    columns = columns_for(kind)
    ws = wb.create_sheet(REVIEW_SHEET)
    for col, (header, width, _) in enumerate(columns, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BORDER
        ws.column_dimensions[cell.column_letter].width = width

    for r, row in enumerate(rows, 2):
        for col, (_, _, key) in enumerate(columns, 1):
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
    ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(columns)).column_letter}{len(rows) + 1}"


def write_summary(wb: Workbook, rows: list[dict], kind: str = "full"):
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

    bold_rows = {1, 5, 7}
    targets = SET_TARGETS.get(kind)
    if targets:
        category_letter = "H"
        top = 8 + len({r["stratum"] for r in rows}) + 1
        bold_rows.add(top)
        ws[f"A{top}"], ws[f"B{top}"], ws[f"C{top}"], ws[f"D{top}"], ws[f"E{top}"] = (
            "Nhóm câu hỏi", "Tổng", "approved", "Mục tiêu", "Còn thiếu (+) / thừa (−)")
        for i, (key, target) in enumerate(targets.items(), top + 1):
            ws[f"A{i}"] = CATEGORY_LABELS.get(key, key)
            ws[f"B{i}"] = f'=COUNTIF({rng(category_letter)},A{i})'
            ws[f"C{i}"] = f'=COUNTIFS({rng(category_letter)},A{i},{rng(status_letter)},"approved")'
            ws[f"D{i}"] = target
            ws[f"E{i}"] = f"=D{i}-C{i}"
    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name=FONT, bold=cell.row in bold_rows)
    ws.column_dimensions["A"].width = 42 if targets else 20


def run(force: bool, kind: str = "full"):
    sheet_path, candidates_path = SETS[kind]["sheet"], SETS[kind]["candidates"]
    if os.path.exists(sheet_path) and not force:
        sys.exit(f"❌ {sheet_path} đã tồn tại — có thể chứa kết quả duyệt dở. Dùng --force để ghi đè.")
    with open(candidates_path, encoding="utf-8") as f:
        candidates = json.load(f)["candidates"]

    rows = build_rows(candidates)
    wb = Workbook()
    wb.remove(wb.active)
    write_guide(wb, kind)
    write_review(wb, rows, kind)
    write_summary(wb, rows, kind)
    wb.active = 1
    # openpyxl không lưu giá trị đã tính của công thức → buộc Excel tính lại khi mở file
    wb.calculation.fullCalcOnLoad = True
    wb.save(sheet_path)
    print(f"✅ {len(rows)} câu → {sheet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ghi đè file Excel đã có")
    parser.add_argument("--set", dest="kind", choices=sorted(SETS), default="full", help="Bộ câu hỏi")
    args = parser.parse_args()
    run(force=args.force, kind=args.kind)
