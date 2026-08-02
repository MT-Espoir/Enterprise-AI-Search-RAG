"""
Test Excel (.xlsx) / CSV ingestion support — table-aware parser (row-group
chunking, tái dùng cơ chế atomic is_table=True có sẵn trong basechunker.py,
xem app/ingestion/parsers/{excel_parser,csv_parser,table_utils}.py).

Chạy: python backend/tests/test_excel_csv_parser.py
"""
import sys, os, tempfile

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openpyxl import Workbook

from app.ingestion.parsers.table_utils import rows_to_markdown_chunks
from app.ingestion.parsers.excel_parser import ExcelParser
from app.ingestion.parsers.csv_parser import CsvParser
from app.ingestion.pipeline import IngestionPipeline
from app.models.document import Document

# ══════════════════════════════════════════════════════════════
# TEST 1: rows_to_markdown_chunks() — gói nhiều dòng thành nhiều chunk vừa đủ
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: rows_to_markdown_chunks() — không nhét cả bảng vào 1 chunk khổng lồ")

header = ["Họ tên", "Chức vụ", "Lương (VNĐ)"]
rows = [[f"Nhân viên số {i}", "Chuyên viên", f"{10_000_000 + i * 100_000}"] for i in range(1, 41)]

chunks = rows_to_markdown_chunks(header, rows, start_row_number=2, max_chars=500, sheet_name="BangLuong")
assert len(chunks) > 1, f"❌ 40 dòng với max_chars=500 phải tách nhiều chunk, chỉ có {len(chunks)}"
for c in chunks:
    assert "Họ tên" in c and "Chức vụ" in c, f"❌ Header phải lặp lại trong mỗi chunk: {c[:80]}"
    assert "[Dòng " in c, f"❌ Thiếu nhãn [Dòng X-Y]: {c[:80]}"
    assert "[Sheet: BangLuong]" in c, f"❌ Thiếu nhãn sheet: {c[:80]}"
print(f"  40 dòng -> {len(chunks)} chunk, mỗi chunk có header + nhãn [Dòng X-Y] ✅")

empty = rows_to_markdown_chunks(header, [], start_row_number=2)
assert empty == [], "❌ Không có dòng dữ liệu phải trả về []"
print("  Không có dòng dữ liệu -> [] ✅")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 2: ExcelParser.parse() — sheet thật qua openpyxl
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: ExcelParser.parse()")

xlsx_path = tempfile.mktemp(suffix=".xlsx")
wb = Workbook()
ws = wb.active
ws.title = "NhanVien"
ws.append(["Họ tên", "Phòng ban", "Lương (VNĐ)"])
for i in range(1, 31):
    ws.append([f"Nhân viên {i}", "Nhân sự", 10_000_000 + i * 150_000])
wb.save(xlsx_path)

try:
    pages = ExcelParser().parse(xlsx_path)
    assert len(pages) == 1, f"❌ 1 sheet phải ra 1 ParsedPage, nhận {len(pages)}"
    page = pages[0]
    assert page.text == "", "❌ page.text phải rỗng (toàn bộ nội dung nằm ở metadata['tables'])"
    assert page.metadata["sheet_name"] == "NhanVien"
    tables = page.metadata["tables"]
    assert len(tables) >= 1, "❌ Phải có ít nhất 1 table chunk"
    assert all("Họ tên" in t for t in tables), "❌ Header phải lặp lại trong mọi chunk"
    print(f"  1 sheet 30 dòng -> {len(tables)} table chunk, header lặp lại đúng ✅")
finally:
    os.remove(xlsx_path)
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 3: CsvParser.parse() — delimiter ";" (kiểu Excel xuất vi-VN)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: CsvParser — auto-detect delimiter ';'")

csv_path = tempfile.mktemp(suffix=".csv")
csv_content = "Họ tên;Phòng ban;Lương (VNĐ)\n" + "\n".join(
    f"Nhân viên {i};Kinh doanh;{9_000_000 + i * 100_000}" for i in range(1, 11)
)
with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
    f.write(csv_content)

try:
    pages = CsvParser().parse(csv_path)
    assert len(pages) == 1
    tables = pages[0].metadata["tables"]
    assert len(tables) >= 1
    # Nếu sniff sai delimiter, mỗi dòng sẽ bị coi là 1 cột duy nhất -> không tách được "Phòng ban"
    assert any("Phòng ban" in t for t in tables), f"❌ Delimiter detect sai, không thấy cột 'Phòng ban': {tables[0][:120]}"
    assert any("Kinh doanh" in t for t in tables), "❌ Dữ liệu cột 'Phòng ban' bị mất/sai tách cột"
    print("  Delimiter ';' auto-detect đúng, tách cột chính xác ✅")
finally:
    os.remove(csv_path)
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 4: CsvParser — fallback encoding cp1258 (không phải UTF-8)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: CsvParser — fallback decode cp1258 (dấu tiếng Việt không bị mất)")

csv_path2 = tempfile.mktemp(suffix=".csv")
# Lưu ý: codec cp1258 built-in của Python có bảng ánh xạ KHÔNG đầy đủ cho các
# ký tự tiếng Việt có 2 dấu tổ hợp (vd "ệ", "ữ", "ộ") — cả dạng NFC lẫn NFD
# đều báo UnicodeEncodeError khi .encode("cp1258") (đã verify trực tiếp), đây
# là giới hạn của thư viện chuẩn Python chứ không phải lỗi của CsvParser. Test
# này vẫn hợp lệ với các từ tiếng Việt dùng dấu ĐƠN (à, ă, đ, â...) — đủ để
# xác nhận nhánh fallback decode hoạt động đúng và không làm mất dấu.
vn_content = "Ho ten,Vi tri\nHoàng Anh,nhân viên\nĐào Văn Nam,nhân viên"
with open(csv_path2, "wb") as f:
    f.write(vn_content.encode("cp1258"))

try:
    pages = CsvParser().parse(csv_path2)
    tables = pages[0].metadata["tables"]
    full_text = "\n".join(tables)
    assert "Hoàng Anh" in full_text, f"❌ Dấu tiếng Việt bị mất khi decode cp1258: {full_text[:200]}"
    assert "Đào Văn Nam" in full_text, f"❌ Dấu tiếng Việt bị mất khi decode cp1258: {full_text[:200]}"
    print("  File cp1258 (không BOM UTF-8) vẫn decode đúng dấu tiếng Việt ✅")
finally:
    os.remove(csv_path2)
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 5: Full IngestionPipeline.ingest() với ExcelParser — chunk có đủ metadata mặc định
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 5: IngestionPipeline.ingest() end-to-end với file .xlsx (embedder/vector_ops giả)")


class FakeEmbedder:
    def embed_batch(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]


class FakeVectorOps:
    def __init__(self):
        self.stored_chunks = []

    def add_chunks(self, chunks):
        self.stored_chunks.extend(chunks)
        return len(chunks)


class FakeDocService:
    def __init__(self):
        self.statuses = {}

    def update_status(self, doc_id, status, **kwargs):
        self.statuses[doc_id] = status


xlsx_path2 = tempfile.mktemp(suffix=".xlsx")
wb2 = Workbook()
ws2 = wb2.active
ws2.title = "BangLuong"
ws2.append(["Họ tên", "Lương"])
ws2.append(["Nhân viên demo", 12_000_000])
wb2.save(xlsx_path2)

vector_ops = FakeVectorOps()
doc_service = FakeDocService()
pipeline = IngestionPipeline(vector_ops=vector_ops, doc_service=doc_service, embedder=FakeEmbedder())

ok = pipeline.ingest(file_path=xlsx_path2, doc_id="doc_excel_1", filename="bangluong.xlsx")
assert ok, "❌ ingest() với file .xlsx phải thành công"
assert doc_service.statuses["doc_excel_1"] == "done"
assert len(vector_ops.stored_chunks) >= 1, "❌ Phải có ít nhất 1 chunk được lưu"

chunk = vector_ops.stored_chunks[0]
meta = chunk["metadata"]
assert meta["document_status"] == Document.DOC_STATUS_CHUA_XAC_DINH, f"❌ Thiếu default document_status: {meta}"
assert meta["document_type"] == "", f"❌ Thiếu default document_type: {meta}"
assert meta["department"] == "", f"❌ Thiếu default department (ACL): {meta}"
assert meta["is_table"] is True, f"❌ Chunk từ Excel phải có is_table=True: {meta}"
assert meta["sheet_name"] == "BangLuong"
print(f"  ingest() .xlsx thành công, {len(vector_ops.stored_chunks)} chunk, đủ default document_status/document_type/department, is_table=True ✅")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 6: Định dạng không hỗ trợ (.xls cũ) -> lỗi rõ ràng, không crash ngầm
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 6: Định dạng .xls (chưa hỗ trợ) -> ValueError rõ ràng")

try:
    pipeline._get_parser("old_file.xls")
    assert False, "❌ .xls chưa hỗ trợ, phải raise ValueError"
except ValueError as e:
    assert "xls" in str(e).lower() or "không được hỗ trợ" in str(e)
    print(f"  _get_parser('old_file.xls') -> ValueError đúng: {e} ✅")
print("  ✅ PASS\n")

print("=" * 60)
print("🎉 TEST EXCEL/CSV PARSER HOÀN THÀNH — TẤT CẢ PASS!")
