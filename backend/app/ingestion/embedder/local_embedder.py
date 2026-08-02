import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

class LocalEmbedder(BaseEmbedder):
    """
    Sử dụng model BGE-M3 (BAAI) chạy local qua LangChain và sentence-transformers.
    Model này hỗ trợ tốt tiếng Việt, đa ngôn ngữ, không cần API Key, chạy hoàn toàn offline.
    """
    
    DEFAULT_MODEL_NAME = "BAAI/bge-m3"

    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        logger.info(f"Khởi tạo Local Embedder với model: {self.model_name} trên {device}...")
        
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True} # Cosine similarity requires normalized embeddings

        # Sử dụng HuggingFaceEmbeddings từ langchain-huggingface (phiên bản mới)
        self.bge_embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info("Khởi tạo Local Embedder hoàn tất.")

    def embed_document(self, text: str) -> List[float]:
        """
        Embed một đoạn text (dành cho document chunk).
        """
        return self.bge_embeddings.embed_documents([text])[0]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed câu hỏi của người dùng. BGE sẽ tối ưu riêng biệt cho query nếu cần.
        """
        return self.bge_embeddings.embed_query(query)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed một danh sách các chunks. 
        Local model hỗ trợ batching siêu tốc qua matrix operation, không lo rate limit.
        """
        return self.bge_embeddings.embed_documents(texts)
