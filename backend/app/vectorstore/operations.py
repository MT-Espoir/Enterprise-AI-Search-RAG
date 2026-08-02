import logging
from .base_vss import BaseVSS
from .client import get_chroma_client, get_collection

logger = logging.getLogger(__name__)


def build_chroma_where(filters: dict) -> dict | None:

    clean = {k: v for k, v in (filters or {}).items() if v is not None}
    if not clean:
        return None
    if len(clean) == 1:
        return clean
    return {"$and": [{k: v} for k, v in clean.items()]}


def build_acl_where(user_department: str | None) -> dict | None:
    """
    Document-level ACL theo phòng ban — CHỈ gọi khi user KHÔNG phải admin
    """
    dept = user_department or ""
    if dept == "":
        return {"department": ""}
    return {"$or": [{"department": ""}, {"department": dept}]}


def combine_where(*clauses: dict | None) -> dict | None:
    present = [c for c in clauses if c]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return {"$and": present}


class VectorStoreOps(BaseVSS):
    """
    Concrete implementation của BaseVSS dùng ChromaDB.
    Tất cả operations đều thao tác trên shared collection duy nhất.
    """

    def __init__(self):
        # Khởi tạo client và lấy collection dùng chung
        client = get_chroma_client()
        self.collection = get_collection(client)
    # WRITE
    def add_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        required = {"id", "text", "embedding", "metadata"}
        for chunk in chunks:
            missing = required - chunk.keys()
            if missing:
                raise ValueError(f"Chunk thiếu keys: {missing} (id={chunk.get('id', 'unknown')})")

        ids        = [c["id"]        for c in chunks]
        documents  = [c["text"]      for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas  = [c["metadata"]  for c in chunks]
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Đã thêm {len(chunks)} chunks vào ChromaDB.")
        return len(chunks)

    # READ
    def query(
        self,
        query_embedding: list[float],
        n_results: int = 20,
        filter_metadata: dict = None,
    ) -> list[dict]:
        """
          distance = 0   → hoàn toàn giống nhau  (similarity = 1.0)
          distance = 2   → hoàn toàn trái ngược  (similarity = -1.0)
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        raw = self.collection.query(**kwargs)

        # ChromaDB trả về list-of-list (1 list per query), lấy index [0]
        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        results = []
        for chunk_id, text, metadata, dist in zip(ids, documents, metadatas, distances):
            results.append({
                "id":       chunk_id,
                "text":     text,
                "metadata": metadata,
                "score":    round(1 - dist, 6),  # Cosine similarity
            })

        return results

    def update_chunks_metadata(self, doc_id: str, metadata_updates: dict) -> int:
        """
        Cập nhật (merge) metadata cho TOÀN BỘ chunk thuộc doc_id
        """
        existing = self.collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        ids = existing["ids"]
        if not ids:
            return 0

        merged_metadatas = []
        for meta in existing["metadatas"]:
            merged = dict(meta)
            merged.update(metadata_updates)
            merged_metadatas.append(merged)

        self.collection.update(ids=ids, metadatas=merged_metadatas)
        logger.info(f"Đã cập nhật metadata cho {len(ids)} chunks của doc_id={doc_id}: {metadata_updates}")
        return len(ids)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Xóa tất cả chunks thuộc về một document.
        Trả về số chunks đã xóa.
        """
        # Trước tiên đếm xem có bao nhiêu chunks sẽ bị xóa
        existing = self.collection.get(where={"doc_id": doc_id})
        count = len(existing["ids"])

        if count > 0:
            self.collection.delete(where={"doc_id": doc_id})
            logger.info(f"Đã xóa {count} chunks của doc_id={doc_id}")

        return count

    def get_collection_count(self) -> int:
        return self.collection.count()

    def chunk_exists(self, chunk_id: str) -> bool:
        result = self.collection.get(ids=[chunk_id])
        return len(result["ids"]) > 0

    def get_first_chunks_of_doc(self, doc_id: str, limit: int = 10, acl_where: dict = None) -> list[dict]:
        """
        Lấy các chunks đầu tiên của một document để phục vụ tính năng Tóm tắt (Summary).
        Bỏ qua semantic search.
        """
        where = {"doc_id": doc_id} if not acl_where else {"$and": [{"doc_id": doc_id}, acl_where]}
        raw = self.collection.get(
            where=where,
            include=["documents", "metadatas"]
        )
        
        if not raw["ids"]:
            return []
            
        chunks = []
        for chunk_id, text, metadata in zip(raw["ids"], raw["documents"], raw["metadatas"]):
            chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
                "score": 1.0  # Điểm tuyệt đối vì đây là fetch trực tiếp
            })
            
        # Sắp xếp lại theo thứ tự chunk_index (để lấy phần mở đầu)
        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

        return chunks[:limit]

    def get_all_chunks(self) -> list[dict]:
        """
        Lấy toàn bộ chunk (text + metadata) của cả collection.
        Dùng để build BM25 index (full-text chỉ tồn tại trong ChromaDB).
        """
        raw = self.collection.get(include=["documents", "metadatas"])

        if not raw["ids"]:
            return []

        return [
            {"id": chunk_id, "text": text, "metadata": metadata}
            for chunk_id, text, metadata in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]
