from datetime import datetime, timezone
from bson import ObjectId
import bcrypt

from .department import DEPARTMENTS, DEFAULT_DEPARTMENT

class User:
    COLLECTION = "users"
    ROLES = ["user", "admin"]

    def __init__(self, username: str, email: str, role: str = "user", department: str = None, password_hash: str = None, _id=None, created_at=None, updated_at=None, last_login=None):
        self._id = _id or ObjectId()
        self.username = username
        self.email = email
        self.role = role if role in self.ROLES else "user"
        # Phòng ban (Document-level ACL) — "" = chưa gán = mở cho mọi tài liệu
        # chưa gán phòng ban. Fallback DEFAULT_DEPARTMENT nếu giá trị không hợp
        # lệ, giống cách `role` fallback "user".
        self.department = department if department in DEPARTMENTS else DEFAULT_DEPARTMENT
        self.password_hash = password_hash

        now = datetime.now(timezone.utc)
        self.created_at = created_at or now
        self.updated_at = updated_at or now
        self.last_login = last_login

    def set_password(self, password: str) -> None:
        """Băm mật khẩu (Hash) bằng bcrypt và lưu lại"""
        salt = bcrypt.gensalt()
        # bcrypt trả về bytes, ta decode sang string để lưu vào MongoDB
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str) -> bool:
        """Kiểm tra mật khẩu nhập vào có khớp với hash không"""
        if not self.password_hash:
            return False
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

    def to_dict(self) -> dict:
        """Chuyển Object thành Dictionary để lưu vào MongoDB hoặc trả qua API (đã ẩn password)"""
        return {
            "_id": self._id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "department": self.department,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_login": self.last_login
        }

    def to_db_dict(self) -> dict:
        """Dictionary đầy đủ (Bao gồm password_hash) để lưu vào DB"""
        data = self.to_dict()
        data["password_hash"] = self.password_hash
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Khởi tạo đối tượng User từ Dictionary lấy dưới MongoDB lên"""
        if not data:
            return None
        return cls(
            _id=data.get("_id"),
            username=data.get("username"),
            email=data.get("email"),
            role=data.get("role", "user"),
            department=data.get("department", DEFAULT_DEPARTMENT),
            password_hash=data.get("password_hash"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            last_login=data.get("last_login")
        )
