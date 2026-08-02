"""
Test OCR cho PDF scan (PDFParser + TesseractOCREngine) — dùng assert thật.

Yêu cầu: đã chạy generate_test_scanned_pdf.py trước, và máy đã cài Tesseract
(kiểm tra qua config.TESSERACT_CMD / mặc định C:\\Program Files\\Tesseract-OCR\\tesseract.exe).
Dùng lang="eng" để test không phụ thuộc việc đã cài traineddata `vie` hay chưa.

Chạy: python backend/tests/test_ocr.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.ocr.tesseract_ocr_engine import TesseractOCREngine

SCANNED_PDF = os.path.join(os.path.dirname(__file__), "test_scanned.pdf")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

if not os.path.exists(SCANNED_PDF):
    print(f"❌ Thiếu {SCANNED_PDF} — chạy generate_test_scanned_pdf.py trước.")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# TEST 1: KHÔNG có ocr_engine — hành vi baseline, trang bị skip (không regression)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: PDFParser không có ocr_engine — trang scan bị bỏ qua")

parser_no_ocr = PDFParser(ocr_engine=None)
pages = parser_no_ocr.parse(SCANNED_PDF)
assert pages == [], f"❌ Không có ocr_engine thì trang scan phải bị bỏ qua, nhận: {pages}"
print("  Không ocr_engine -> pages=[] (đúng hành vi baseline): ✅")
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
# TEST 2: CÓ ocr_engine (Tesseract thật, lang=eng) — trang được phục hồi
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: PDFParser với TesseractOCREngine thật — phục hồi text")

ocr_engine = TesseractOCREngine(tesseract_cmd=TESSERACT_CMD, languages="eng", dpi=200)
parser_with_ocr = PDFParser(ocr_engine=ocr_engine)
pages = parser_with_ocr.parse(SCANNED_PDF)

assert len(pages) == 1, f"❌ Phải phục hồi đúng 1 trang, nhận: {len(pages)}"
page = pages[0]
print(f"  OCR text: {page.text[:200]!r}")

assert page.metadata["is_scanned"] is True, f"❌ is_scanned phải True, nhận: {page.metadata['is_scanned']}"
assert page.metadata["ocr_used"] is True, f"❌ ocr_used phải True, nhận: {page.metadata['ocr_used']}"
print("  is_scanned=True, ocr_used=True: ✅")

text_lower = page.text.lower()
assert "sensormqttlistener" in text_lower or "sensor" in text_lower, \
    f"❌ OCR text không chứa từ khóa mong đợi: {page.text!r}"
print("  OCR text chứa từ khóa đặc trưng ('sensor'): ✅")
print("  ✅ PASS\n")

print("=" * 60)
print("🎉 TẤT CẢ TESTS PASS!")
