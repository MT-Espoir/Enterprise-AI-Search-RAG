import logging

from openpyxl import load_workbook

from .base_parser import BaseParser, ParsedPage
from .table_utils import rows_to_markdown_chunks

logger = logging.getLogger(__name__)


class ExcelParser(BaseParser):
    def parse(self, file_path: str) -> list[ParsedPage]:
        pages = []

        wb = load_workbook(file_path, data_only=True, read_only=True)
        try:
            for sheet_idx, ws in enumerate(wb.worksheets, start=1):
                rows_iter = ws.iter_rows(values_only=True)
                try:
                    header_row = next(rows_iter)
                except StopIteration:
                    continue  # sheet rỗng hoàn toàn

                header = [self.clean_text(str(c)) if c is not None else "" for c in header_row]
                if not any(header):
                    continue  # dòng đầu rỗng -> không có gì để làm header

                data_rows = [
                    [self.clean_text(str(c)) if c is not None else "" for c in row]
                    for row in rows_iter
                    if any(cell is not None and str(cell).strip() for cell in row)
                ]
                if not data_rows:
                    continue  # chỉ có header, không có dữ liệu

                tables = rows_to_markdown_chunks(
                    header, data_rows, start_row_number=2, sheet_name=ws.title
                )

                pages.append(
                    ParsedPage(
                        page_num=sheet_idx,
                        text="",
                        metadata={
                            "tables": tables,
                            "sheet_name": ws.title,
                            "source": "excel",
                            "char_count": sum(len(t) for t in tables),
                        },
                    )
                )
        finally:
            wb.close()

        return pages
