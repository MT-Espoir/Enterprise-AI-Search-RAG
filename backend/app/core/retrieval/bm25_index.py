import hashlib
import re
import threading

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def simple_tokenize(text: str) -> list[str]:
    """
    Tokenizer đơn giản cho BM25: lowercase + tách theo \\w+ (Unicode-aware).
    """
    return _TOKEN_RE.findall(text.lower())


def _compute_checksum(chunks: list[dict]) -> str:
    """Checksum để quyết định có rebuild BM25 hay không."""
    parts = []
    for c in chunks:
        m = c.get("metadata", {}) or {}
        parts.append(
            f'{c["id"]}\x1f{m.get("department", "")}\x1f'
            f'{m.get("document_status", "")}\x1f{m.get("document_type", "")}'
        )
    joined = "|".join(sorted(parts))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


class BM25Index:

    def __init__(self):
        self._bm25 = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._checksum = None
        self._lock = threading.Lock()

    def build(self, chunks: list[dict], force: bool = False) -> bool:
        """
        Rebuild toàn bộ index từ danh sách chunk (format get_all_chunks()).
        Trả về False nếu checksum không đổi (bỏ qua rebuild tốn kém), True nếu đã rebuild.
        """
        checksum = _compute_checksum(chunks)
        if not force and checksum == self._checksum:
            return False

        tokenized = [simple_tokenize(c["text"]) for c in chunks]
        new_bm25 = BM25Okapi(tokenized) if tokenized else None

        with self._lock:
            self._bm25 = new_bm25
            self._checksum = checksum
            self._ids = [c["id"] for c in chunks]
            self._texts = [c["text"] for c in chunks]
            self._metadatas = [c["metadata"] for c in chunks]

        return True

    def refresh(self, vector_ops) -> bool:
        """Convenience: rebuild trực tiếp từ VectorStoreOps.get_all_chunks()."""
        return self.build(vector_ops.get_all_chunks())

    def search(self, question: str, top_k: int = 20, doc_id: str = None, filters: dict = None,
               acl_department: str = None, acl_bypass: bool = False) -> list[dict]:
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(simple_tokenize(question))
        indices = range(len(scores))

        combined_filters = dict(filters or {})
        if doc_id:
            combined_filters["doc_id"] = doc_id
        if combined_filters:
            indices = [
                i for i in indices
                if all(self._metadatas[i].get(k) == v for k, v in combined_filters.items())
            ]

        if not acl_bypass:
            user_dept = acl_department or ""
            indices = [i for i in indices if self._metadatas[i].get("department", "") in ("", user_dept)]

        ranked = sorted(indices, key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {
                "id": self._ids[i],
                "text": self._texts[i],
                "metadata": self._metadatas[i],
                "score": float(scores[i]),
            }
            for i in ranked
        ]


_shared_instance: "BM25Index | None" = None
_shared_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    global _shared_instance
    if _shared_instance is None:
        with _shared_lock:
            if _shared_instance is None:
                _shared_instance = BM25Index()
    return _shared_instance
