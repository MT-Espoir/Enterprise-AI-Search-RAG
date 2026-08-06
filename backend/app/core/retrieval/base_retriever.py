from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import RetrievedChunk


class BaseRetriever(ABC):
    ops = None

    @abstractmethod
    def retrieve(
        self,
        question: str,
        doc_id: str = None,
        history: list[dict] = None,
        vector_query: str = None,
        filters: dict = None,
        acl_department: str = None,
        acl_bypass: bool = False,
    ) -> list["RetrievedChunk"]:
        ...
