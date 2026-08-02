"""
Tạo test_scanned.pdf — 1 trang PDF KHÔNG có text layer thật (chỉ có 1 ảnh nhúng),
dùng để test OCR. Text trong ảnh cố tình dùng tiếng Anh để test không phụ thuộc
việc đã cài traineddata `vie` cho Tesseract hay chưa.

Chạy: python backend/tests/generate_test_scanned_pdf.py
"""
import os

import fitz
from PIL import Image, ImageDraw, ImageFont

OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "test_scanned.pdf")
OUTPUT_PNG = os.path.join(os.path.dirname(__file__), "test_scanned_page.png")

TEXT = (
    "SCANNED DOCUMENT TEST\n\n"
    "SensorMQTTListener subscribes to MQTT topic tele plus SENSOR.\n"
    "This page has no real text layer, only an embedded image.\n"
    "Used to verify Tesseract OCR fallback in PDFParser."
)


def _load_font(size: int):
    for candidate in ("arial.ttf", "calibri.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default(size=size)


def generate():
    img = Image.new("RGB", (1240, 800), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(36)
    draw.multiline_text((60, 60), TEXT, fill="black", font=font, spacing=14)
    img.save(OUTPUT_PNG)

    doc = fitz.open()
    page = doc.new_page(width=1240, height=800)
    page.insert_image(page.rect, filename=OUTPUT_PNG)
    doc.save(OUTPUT_PDF)
    doc.close()

    print(f"Generated scanned test PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    generate()
