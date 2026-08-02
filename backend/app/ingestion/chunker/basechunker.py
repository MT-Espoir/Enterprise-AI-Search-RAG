from abc import ABC, abstractmethod
from typing import Any


class BaseChunker(ABC):
    @abstractmethod
    def chunk_text(self, text: str) -> list[str]:
        """Chunk một đoạn text."""
        pass

    def chunk_pages(self, pages: list[Any]) -> list[dict]:
        """
        Chunk toàn bộ document và tách riêng Table Chunk.
        """
        all_chunks = []
        for page in pages:
            # 1. Chunk normal text
            page_chunks = self.chunk_text(page.text)
            
            # Lấy metadata và tách danh sách tables ra
            base_metadata = dict(getattr(page, "metadata", {}) or {})
            tables = base_metadata.pop("tables", [])
            
            # Thêm chunks văn bản thông thường
            for idx, chunk in enumerate(page_chunks):
                chunk_meta = base_metadata.copy()
                chunk_meta["page_num"] = page.page_num
                chunk_meta["chunk_index"] = idx
                chunk_meta["is_table"] = False
                all_chunks.append(
                    {
                        "text": chunk,
                        "metadata": chunk_meta,
                    }
                )
            
            # 2. Thêm các Table chunks nguyên vẹn
            for idx, table_md in enumerate(tables):
                table_meta = base_metadata.copy()
                table_meta["page_num"] = page.page_num
                table_meta["chunk_index"] = len(page_chunks) + idx
                table_meta["is_table"] = True
                all_chunks.append(
                    {
                        "text": table_md,
                        "metadata": table_meta,
                    }
                )
                
        return all_chunks