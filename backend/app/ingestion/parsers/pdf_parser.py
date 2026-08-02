import logging

import fitz
from .base_parser import BaseParser, ParsedPage

logger = logging.getLogger(__name__)

class PDFParser(BaseParser):
    MIN_TEXT_LENGTH = 50  # Bỏ qua trang có ít hơn 50 ký tự

    def __init__(self, ocr_engine=None):
        """
        Args:
            ocr_engine: (Tùy chọn) object có method extract_text(page) -> str,
                        vd TesseractOCREngine. Nếu None, hành vi giữ nguyên như
                        trước khi có OCR (bỏ qua trang thiếu text).
        """
        self.ocr_engine = ocr_engine
        self.ocr_stats = {
            "pages_total": 0,
            "pages_needing_ocr": 0,
            "pages_ocr_attempted": 0,
            "pages_ocr_failed": 0,
        }

    def extract_spans(self, page, table_bboxes=None):
        table_bboxes = table_bboxes or []
        text_dict = page.get_text("dict")

        for block in text_dict["blocks"]:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    span_rect = fitz.Rect(span["bbox"])

                    # Kiểm tra xem span có nằm trong bất kỳ bảng nào không
                    is_in_table = False
                    for tb_bbox in table_bboxes:
                        # Dùng giao nhau (intersects) để xác định text thuộc bảng
                        if span_rect.intersects(tb_bbox):
                            is_in_table = True
                            break

                    if is_in_table:
                        continue

                    yield {
                        "text": span["text"],
                        "bbox": span["bbox"],
                        "font": span["font"],
                        "size": span["size"],
                        "flags": span["flags"],
                        "color": span["color"],
                    }

    def _is_scanned(self, spans) -> bool:
        return len(spans) == 0

    def _try_ocr(self, page, page_num: int) -> str:
        """Thử OCR trang khi thiếu text layer. Fail-open: không có ocr_engine hoặc
        OCR lỗi đều trả về chuỗi rỗng (hành vi tương đương trước khi có OCR)."""
        self.ocr_stats["pages_needing_ocr"] += 1
        if not self.ocr_engine:
            logger.warning(
                f"Trang {page_num}: không có text layer/bảng và OCR đang tắt hoặc "
                f"chưa cấu hình — bỏ qua trang này."
            )
            return ""
        self.ocr_stats["pages_ocr_attempted"] += 1
        return self.clean_text(self.ocr_engine.extract_text(page))

    def parse(self, file_path: str) -> list[ParsedPage]:

        pages = []

        with fitz.open(file_path) as doc:

            for page_num, page in enumerate(doc, start=1):
                self.ocr_stats["pages_total"] += 1

                # 1. Trích xuất bảng (Tables)
                table_finder = page.find_tables()
                tables_md = []
                table_bboxes = []

                if table_finder and table_finder.tables:
                    for table in table_finder.tables:
                        # Lấy bounding box để filter
                        table_bboxes.append(fitz.Rect(table.bbox))
                        # Chuyển thành markdown
                        tables_md.append(table.to_markdown())

                # 2. Trích xuất text thông thường (loại trừ text trong bảng)
                spans = list(self.extract_spans(page, table_bboxes))
                if not spans and not tables_md:
                    had_ocr_engine = self.ocr_engine is not None
                    ocr_text = self._try_ocr(page, page_num)
                    if len(ocr_text) < self.MIN_TEXT_LENGTH:
                        if had_ocr_engine:
                            self.ocr_stats["pages_ocr_failed"] += 1
                        continue

                    pages.append(
                        ParsedPage(
                            page_num=page_num,
                            text=ocr_text,
                            metadata={
                                "has_text": False,
                                "has_images": bool(page.get_images(full=True)),
                                "char_count": len(ocr_text),
                                "is_scanned": True,
                                "ocr_used": True,
                                "tables": [],
                            },
                        )
                    )
                    continue

                texts = [
                    cleaned
                    for span in spans
                    if (cleaned := self.clean_text(span["text"]))
                ]

                text = " ".join(texts)
                ocr_used = False
                if len(text) < self.MIN_TEXT_LENGTH and not tables_md:
                    had_ocr_engine = self.ocr_engine is not None
                    ocr_text = self._try_ocr(page, page_num)
                    if len(ocr_text) > len(text):
                        text = ocr_text
                        ocr_used = True
                    if len(text) < self.MIN_TEXT_LENGTH:
                        if had_ocr_engine:
                            self.ocr_stats["pages_ocr_failed"] += 1
                        continue

                pages.append(
                    ParsedPage(
                        page_num=page_num,
                        text=text,
                        metadata={
                            "has_text": len(spans) > 0,
                            "has_images": bool(page.get_images(full=True)),
                            "char_count": len(text),
                            "is_scanned": self._is_scanned(spans) and not tables_md,
                            "ocr_used": ocr_used,
                            "tables": tables_md,  # Thêm tables markdown vào metadata
                        },
                    )
                )

        return pages
