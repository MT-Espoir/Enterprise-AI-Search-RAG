import io
import logging

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class TesseractOCREngine:
    """
    OCR trang PDF không có text layer (scan thật) qua Tesseract (subprocess, không
    load model vào RAM Python — không cần singleton/getter như BM25Index/Reranker).

    Fail-open: bất kỳ lỗi nào (thiếu binary, thiếu traineddata ngôn ngữ, lỗi decode
    ảnh...) đều trả về chuỗi rỗng + log warning.
    """

    def __init__(self, tesseract_cmd: str = None, languages: str = "vie+eng", dpi: int = 200):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.languages = languages
        self.dpi = dpi

    def extract_text(self, page) -> str:
        try:
            pix = page.get_pixmap(dpi=self.dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img, lang=self.languages).strip()
        except Exception as exc:
            logger.warning(f"OCR thất bại trang {page.number + 1}: {exc} — bỏ qua, không có text cho trang này.")
            return ""
