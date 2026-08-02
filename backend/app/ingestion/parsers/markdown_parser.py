import re
from .base_parser import BaseParser, ParsedPage

class MarkdownParser(BaseParser):
    # Regex khớp với cú pháp Markdown cần loại bỏ
    _MD_STRIP = [
        re.compile(r'\[([^\]]+)\]\([^)]+\)'),  # [text](url)  → text
        re.compile(r'`{3}.*?`{3}', re.DOTALL),  # ```code block``` → xóa
        re.compile(r'`([^`]+)`'),               # `inline code`    → text
        re.compile(r'^#{1,6}\s+', re.MULTILINE), # # Heading        → xóa dấu #
        re.compile(r'\*{1,2}([^*]+)\*{1,2}'),   # **bold** / *em*  → text
        re.compile(r'^[-*+] ', re.MULTILINE),    # - bullet         → xóa dấu gạch
        re.compile(r'^>{1,} ?', re.MULTILINE),   # > blockquote     → xóa
    ]

    def _md_to_plain_text(self, text: str) -> str:
        for pattern in self._MD_STRIP:
            text = pattern.sub(lambda m: m.group(1) if m.lastindex else '', text)
        return self.clean_text(text)

    def parse(self, file_path: str) -> list[ParsedPage]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        plain_text = self._md_to_plain_text(raw)

        if not plain_text:
            return []

        return [
            ParsedPage(
                page_num=1,
                text=plain_text,
                metadata={
                    "char_count": len(plain_text),
                    "source": "markdown",
                }
            )
        ]
