from datetime import datetime, timezone
from dataclasses import dataclass, field
from bson import ObjectId

@dataclass
class SourceRef:
    doc_id: str
    filename: str
    chunk_text: str
    score: float
    page: int = None

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "chunk_text": self.chunk_text,
            "score": self.score,
            "page": self.page
        }

@dataclass
class Message:
    role: str  # "user" hoặc "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sources: list[SourceRef] = field(default_factory=list)
    tokens_used: int = None
    # Tier 1 Security Guardrails (xem app/core/guardrails/) — {"sanitized_chunks":
    # int, "citation_warning": bool, "unverified_citations": [...]}. None cho
    # message role="user" (guardrail chỉ áp dụng cho câu trả lời assistant).
    guardrail: dict = None

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "sources": [s.to_dict() for s in self.sources] if self.sources else [],
            "tokens_used": self.tokens_used,
            "guardrail": self.guardrail
        }

class ChatSession:
    COLLECTION = "chat_sessions"

    def __init__(self, user_id: str, title: str = "", _id=None, messages=None, message_count=0, created_at=None, updated_at=None):
        self._id = _id or ObjectId()
        self.user_id = str(user_id)
        self.title = title
        
        # messages không phải lúc nào cũng được load đầy đủ (để tiết kiệm RAM)
        self.messages = messages or []
        self.message_count = message_count
        
        now = datetime.now(timezone.utc)
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def add_message(self, role: str, content: str, sources=None, tokens_used=None, guardrail=None):
        msg = Message(role=role, content=content, sources=sources or [], tokens_used=tokens_used, guardrail=guardrail)
        self.messages.append(msg)
        self.message_count += 1
        self.updated_at = datetime.now(timezone.utc)
        return msg

    def to_dict(self) -> dict:
        return {
            "_id": self._id,
            "user_id": self.user_id,
            "title": self.title,
            "message_count": self.message_count,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatSession":
        if not data:
            return None
            
        # Parse messages
        parsed_messages = []
        for m_data in data.get("messages", []):
            sources = [SourceRef(**s) for s in m_data.get("sources", [])]
            msg = Message(
                role=m_data.get("role"),
                content=m_data.get("content"),
                timestamp=m_data.get("timestamp"),
                sources=sources,
                tokens_used=m_data.get("tokens_used"),
                guardrail=m_data.get("guardrail")
            )
            parsed_messages.append(msg)
            
        return cls(
            _id=data.get("_id"),
            user_id=data.get("user_id"),
            title=data.get("title", ""),
            messages=parsed_messages,
            message_count=data.get("message_count", len(parsed_messages)),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )
