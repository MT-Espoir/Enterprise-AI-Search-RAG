from langchain_text_splitters import RecursiveCharacterTextSplitter

from .basechunker import BaseChunker


class RecursiveChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_length: int = 50,
    ):
        self.min_chunk_length = min_chunk_length
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                r"\n(?=Chương\s+[IVXLCDM]+\b)",
                r"\n(?=Điều\s+\d+)",
                "\n\n",
                "\n",
                " ",
                "",
            ],
            is_separator_regex=True,
        )
    def chunk_text(self, text: str) -> list[str]:
        chunks = self.splitter.split_text(text)
        return [
            chunk.strip()
            for chunk in chunks
            if len(chunk.strip()) >= self.min_chunk_length
        ]   