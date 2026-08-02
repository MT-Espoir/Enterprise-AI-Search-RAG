import logging
from sentence_transformers import CrossEncoder
from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)

class Reranker:
    _instance = None
    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model_name: str = DEFAULT_MODEL, top_k: int = 5):
        logger.info(f"Đang load Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
        self.top_k = top_k
        logger.info("Cross-Encoder model đã sẵn sàng.")

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL, top_k: int = 5) -> "Reranker":
        """
        Trả về Singleton instance.
        Dùng get_instance() thay vì Reranker() trực tiếp để tránh load model nhiều lần.
        """
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, top_k=top_k)
        return cls._instance

    def rerank(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Re-rank các chunks dựa trên Cross-Encoder score.

        Args:
            question: Câu hỏi gốc của người dùng.
            chunks:   Danh sách chunks từ Retriever (top ~20).

        Returns:
            Top self.top_k chunks sau khi rerank, sắp xếp theo score giảm dần.
        """
        if not chunks:
            return []

        pairs = [(question, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.metadata["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c.metadata["rerank_score"], reverse=True)
        return reranked[:self.top_k]
