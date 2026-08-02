"""
Helper dùng chung cho ExcelParser/CsvParser — gói dữ liệu dạng bảng (hàng/cột)
thành nhiều chunk markdown, mỗi chunk LẶP LẠI header + có nhãn [Dòng X-Y] để
LLM/user biết vị trí gốc trong file.
"""


def _row_to_markdown_line(row: list[str]) -> str:
    escaped = [str(cell if cell is not None else "").replace("|", "\\|").replace("\n", " ") for cell in row]
    return "| " + " | ".join(escaped) + " |"


def rows_to_markdown_chunks(
    header: list[str],
    rows: list[list],
    start_row_number: int = 2,
    max_chars: int = 1000,
    sheet_name: str = None,
) -> list[str]:
    """Gói tham lam (greedy) các dòng dữ liệu theo ngân sách ký tự — KHÔNG cắt
    ngang 1 dòng, luôn giữ ít nhất 1 dòng/chunk kể cả khi 1 dòng đã vượt
    max_chars một mình. `start_row_number` = số thứ tự dòng dữ liệu ĐẦU TIÊN
    trong file gốc (mặc định 2 vì dòng 1 là header) — dùng để gắn nhãn
    [Dòng X-Y] giúp người dùng đối chiếu lại file gốc."""
    if not rows:
        return []

    header_line = _row_to_markdown_line(header)
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
    header_block = header_line + "\n" + separator_line
    label_prefix = f"[Sheet: {sheet_name}] " if sheet_name else ""

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = len(header_block)
    group_start = start_row_number

    def flush(end_row_number: int) -> None:
        nonlocal current_lines, current_len
        if not current_lines:
            return
        label = f"{label_prefix}[Dòng {group_start}-{end_row_number}]"
        body = "\n".join(current_lines)
        chunks.append(f"{label}\n{header_block}\n{body}")
        current_lines = []
        current_len = len(header_block)

    for i, row in enumerate(rows):
        row_number = start_row_number + i
        line = _row_to_markdown_line(row)
        added_len = len(line) + 1  # +1 cho ký tự nối dòng
        if current_lines and current_len + added_len > max_chars:
            flush(row_number - 1)
            group_start = row_number
        current_lines.append(line)
        current_len += added_len

    flush(start_row_number + len(rows) - 1)
    return chunks
