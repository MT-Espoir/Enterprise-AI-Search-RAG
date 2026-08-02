from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
import unicodedata

@dataclass
class ParsedPage:
    """Class đại diện cho một trang tài liệu sau khi đọc xong"""
    page_num: int      # Số trang (bắt đầu từ 1)
    text: str          # Nội dung chữ trên trang
    metadata: dict = field(default_factory=dict)  # Khởi tạo dict rỗng mặc định nếu không truyền vào

class BaseParser(ABC):

    MULTISPACE = re.compile(r"[ \t]+")
    MULTINEWLINE = re.compile(r"\n{3,}")
    HYPHEN_BREAK = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")

    @abstractmethod
    def parse(self, file_path: str) -> list[ParsedPage]:

        pass

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)

        text = (
            text.replace("\u00ad", "")   # soft hyphen
                .replace("\ufeff", "")
                .replace("\u200b", "")
        )

        text = self.HYPHEN_BREAK.sub("", text)

        text = self.MULTISPACE.sub(" ", text)

        text = self.MULTINEWLINE.sub("\n\n", text)
        return text.strip()