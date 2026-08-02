"""
CLI script: tạo tài khoản admin đầu tiên.

Cần thiết vì /register công khai đã bị đóng (chỉ admin mới tạo được user mới
qua API /api/users) — nhưng lúc DB chưa có admin nào thì không ai gọi được
API đó cả (bài toán con gà quả trứng). Script này bootstrap bằng cách gọi
thẳng AuthService, không qua HTTP.

Chạy:
    python backend/scripts/create_admin.py <email> <username> <password>
"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.services.auth_service import AuthService, AuthError
from app.models.user import User
from app.extensions import mongo
from bson import ObjectId


def main():
    if len(sys.argv) != 4:
        print("Cách dùng: python backend/scripts/create_admin.py <email> <username> <password>")
        sys.exit(1)

    email, username, password = sys.argv[1], sys.argv[2], sys.argv[3]

    app = create_app("development")
    with app.app_context():
        auth_service = AuthService()
        try:
            user = auth_service.register(username, email, password)
        except AuthError as e:
            print(f"❌ {e.message}")
            sys.exit(1)

        mongo.db[User.COLLECTION].update_one(
            {"_id": ObjectId(user["_id"])}, {"$set": {"role": "admin"}}
        )
        print(f"✅ Đã tạo admin: {username} ({email}), user_id={user['_id']}")


if __name__ == "__main__":
    main()
