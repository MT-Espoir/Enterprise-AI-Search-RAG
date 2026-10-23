import chromadb
from flask import current_app

_client = None  # Singleton — chỉ tạo 1 lần duy nhất trong vòng đời app
_ef_search_applied = False  # ef_search chỉ cần áp 1 lần mỗi tiến trình, xem get_collection()

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

    ef_search (CHROMA_HNSW_EF_SEARCH): mặc định của collection là 100 — đo 2026-10-24 trên
    13.571 chunk, HNSW bỏ sót HẲN tài liệu đúng ở vài câu hỏi Excel (ANN recall@10 0.965);
    500 → 0.998, chỉ thêm ~1ms/truy vấn. ef_search được CHỐT khi index nạp ở truy vấn đầu
    tiên của tiến trình → áp ngay lần lấy collection đầu tiên (trước mọi truy vấn); đổi giá
    trị cần khởi động lại server (CHROMA_MODE=http: khởi động lại cả Chroma server).
    """
    global _ef_search_applied
    name = current_app.config.get("CHROMA_COLLECTION_NAME", "enterprise_documents")
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )
    ef_search = current_app.config.get("CHROMA_HNSW_EF_SEARCH")
    if ef_search and not _ef_search_applied:
        current = ((collection.configuration or {}).get("hnsw") or {}).get("ef_search")
        if current != ef_search:
            collection.modify(configuration={"hnsw": {"ef_search": ef_search}})
            collection = client.get_collection(name)
        _ef_search_applied = True
    return collection
