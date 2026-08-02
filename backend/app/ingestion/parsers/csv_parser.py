import csv
import io
import logging

from .base_parser import BaseParser, ParsedPage
from .table_utils import rows_to_markdown_chunks

logger = logging.getLogger(__name__)

_ENCODINGS_TRY_ORDER = ("utf-8-sig", "cp1258", "latin-1")


class CsvParser(BaseParser):
    def parse(self, file_path: str) -> list[ParsedPage]:
        with open(file_path, "rb") as f:
            raw = f.read()

        text = None
        used_encoding = None
        for enc in _ENCODINGS_TRY_ORDER:
            try:
                text = raw.decode(enc)
                used_encoding = enc
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            # latin-1 map 1-1 byte->codepoint nên về lý thuyết không bao giờ
            # decode lỗi — nhánh này chỉ để fail-safe tuyệt đối.
            text = raw.decode("utf-8", errors="replace")
            used_encoding = "utf-8 (replace ký tự lỗi)"

        if used_encoding != "utf-8-sig":
            logger.warning(
                f"CSV không phải UTF-8 (có BOM) — đã fallback decode bằng '{used_encoding}'. "
                f"Nếu dấu tiếng Việt hiển thị sai, hãy lưu lại file dạng UTF-8."
            )

        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","  # không đoán được -> mặc định CSV chuẩn

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            return []

        header = [self.clean_text(c) for c in rows[0]]
        if not any(header):
            return []

        data_rows = [[self.clean_text(c) for c in row] for row in rows[1:]]
        if not data_rows:
            return []

        tables = rows_to_markdown_chunks(header, data_rows, start_row_number=2)

        return [
            ParsedPage(
                page_num=1,
                text="",
                metadata={
                    "tables": tables,
                    "source": "csv",
                    "char_count": sum(len(t) for t in tables),
                },
            )
        ]
