import chromadb
from flask import current_app

_client = None  # Singleton — chỉ tạo 1 lần duy nhất trong vòng đời app

def get_chroma_client() -> chromadb.ClientAPI:
    """
      - CHROMA_MODE = "http"        → HttpClient  (kết nối server riêng, dùng khi Docker)
      - CHROMA_MODE = "persistent"  → PersistentClient (lưu local, dùng khi dev)
    """
    global _client
    if _client is None:
        mode = current_app.config.get("CHROMA_MODE", "http")

        if mode == "persistent":
            path = current_app.config.get("CHROMA_PERSIST_PATH", "./chromadb_data")
            _client = chromadb.PersistentClient(path=path)
        else:
            host = current_app.config.get("CHROMA_HOST", "localhost")
            port = int(current_app.config.get("CHROMA_PORT", 8000))
            _client = chromadb.HttpClient(host=host, port=port)

    return _client

def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """
    Lấy hoặc tạo collection dùng chung (shared collection).
    metadata hnsw:space = cosine → ChromaDB dùng Cosine distance thay vì Euclidean.
    """
    name = current_app.config.get("CHROMA_COLLECTION_NAME", "enterprise_documents")
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )
