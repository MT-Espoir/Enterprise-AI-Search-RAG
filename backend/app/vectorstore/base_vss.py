from abc import ABC, abstractmethod
from typing import Any

class BaseVSS(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[dict]) -> int:
        pass
    @abstractmethod
    def query(self, query_embedding: list[float], n_results: int = 20, filter_metadata: dict = None) -> list[dict]:
        pass
    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> int:
        pass
    @abstractmethod
    def get_collection_count(self) -> int:
        pass
    @abstractmethod
    def chunk_exists(self, chunk_id: str) -> bool:
        pass
        