from google import genai
from typing import List
from .base_embedder import BaseEmbedder
import time
class GoogleEmbedder(BaseEmbedder):
    DEFAULT_MODEL = "models/gemini-embedding-2"

    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_document(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config={
                "task_type": "RETRIEVAL_DOCUMENT"
            }
        )
        return result.embeddings[0].values

    def embed_query(self, query: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=query,
            config={
                "task_type": "RETRIEVAL_QUERY"
            }
        )
        return result.embeddings[0].values

    def embed_batch(self, texts: List[str], delay_ms: int = 100) -> List[List[float]]:
        embedding = []
        for i, text in enumerate(texts):
            embedding.append(self._embed_with_retry(text, task_type="RETRIEVAL_DOCUMENT"))
            if i < len(texts) -1 :
                time.sleep(delay_ms/1000)
        return embedding
    
    def _embed_with_retry(self, text: str, task_type: str, max_retries: int = 3) -> list[float]:
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config={"task_type": task_type}
                )
                return result.embeddings[0].values
            except Exception as e:
                if attempt == max_retries - 1:
                    raise                         
                wait = 2 ** (attempt + 1)         # 2, 4, 8 giây
                print(f"Retry {attempt+1} sau {wait}s do lỗi: {e}")
                time.sleep(wait)