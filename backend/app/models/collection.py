from datetime import datetime, timezone
from bson import ObjectId

class Collection:
    COLLECTION = "chroma_collections"

    def __init__(self, name: str, description: str = "", embedding_model: str = "", _id=None, document_count=0, chunk_count=0, created_at=None, updated_at=None):
        self._id = _id or ObjectId()
        self.name = name
        self.description = description
        self.embedding_model = embedding_model
        self.document_count = document_count
        self.chunk_count = chunk_count
        
        now = datetime.now(timezone.utc)
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self) -> dict:
        return {
            "_id": self._id,
            "name": self.name,
            "description": self.description,
            "embedding_model": self.embedding_model,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Collection":
        if not data:
            return None
        return cls(
            _id=data.get("_id"),
            name=data.get("name"),
            description=data.get("description", ""),
            embedding_model=data.get("embedding_model", ""),
            document_count=data.get("document_count", 0),
            chunk_count=data.get("chunk_count", 0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )
