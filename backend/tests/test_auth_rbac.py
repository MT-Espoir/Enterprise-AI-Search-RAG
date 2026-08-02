"""
Test JWT auth + RBAC (admin/nhân viên) — dùng Flask test client + MongoDB
test DB (`rag_test_db`, TestingConfig có sẵn), KHÔNG đụng `rag_db` thật.

ChromaDB cũng được trỏ sang thư mục tạm riêng (env CHROMA_PERSIST_PATH override
TRƯỚC khi tạo app) — cách ly hoàn toàn khỏi `chromadb_data` thật đang được
batch ingest sử dụng đồng thời, tránh xung đột ghi ChromaDB giữa 2 process.

Chạy: python backend/tests/test_auth_rbac.py
"""
import sys, os, tempfile, shutil

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_tmp_chroma = tempfile.mkdtemp(prefix="rag_test_chroma_")
os.environ["CHROMA_PERSIST_PATH"] = _tmp_chroma

from bson import ObjectId

from app import create_app
from app.extensions import mongo
from app.services.auth_service import AuthService
from app.models.user import User
from app.models.document import Document
from app.models.chat_session import ChatSession


def clean_db():
    mongo.db[User.COLLECTION].delete_many({})
    mongo.db[Document.COLLECTION].delete_many({})
    mongo.db[ChatSession.COLLECTION].delete_many({})


def make_user(auth_service, username, email, password, role):
    user = auth_service.register(username, email, password)
    if role != "user":
        mongo.db[User.COLLECTION].update_one({"_id": ObjectId(user["_id"])}, {"$set": {"role": role}})
    login = auth_service.login(email, password)
    return user["_id"], login["access_token"]


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}


app = create_app("testing")
client = app.test_client()

with app.app_context():
    clean_db()
    auth_service = AuthService()

    admin_id, admin_token = make_user(auth_service, "admin1", "admin1@test.local", "pass1234", "admin")
    emp1_id, emp1_token = make_user(auth_service, "emp1", "emp1@test.local", "pass1234", "user")
    emp2_id, emp2_token = make_user(auth_service, "emp2", "emp2@test.local", "pass1234", "user")

    print("=" * 60)
    print("TEST 1: JWT chứa đúng claim role")
    import jwt as pyjwt
    decoded_admin = pyjwt.decode(admin_token, options={"verify_signature": False})
    decoded_emp = pyjwt.decode(emp1_token, options={"verify_signature": False})
    assert decoded_admin.get("role") == "admin", f"❌ JWT admin thiếu role=admin: {decoded_admin}"
    assert decoded_emp.get("role") == "user", f"❌ JWT nhân viên thiếu role=user: {decoded_emp}"
    print("  ✅ PASS\n")

    print("=" * 60)
    print("TEST 2: Gọi API không có token -> 401")
    r = client.get("/api/documents/")
    assert r.status_code == 401, f"❌ Kỳ vọng 401, nhận {r.status_code}: {r.get_json()}"
    r = client.post("/api/chat/message", json={"message": "hi"})
    assert r.status_code == 401, f"❌ Kỳ vọng 401, nhận {r.status_code}"
    print("  ✅ PASS\n")

    print("=" * 60)
    print("TEST 3: Nhân viên gọi endpoint admin_required -> 403, admin -> 200")
    r = client.get("/api/collections/", headers=auth_header(emp1_token))
    assert r.status_code == 403, f"❌ Kỳ vọng 403, nhận {r.status_code}: {r.get_json()}"
    r = client.get("/api/collections/", headers=auth_header(admin_token))
    assert r.status_code == 200, f"❌ Kỳ vọng 200, nhận {r.status_code}: {r.get_json()}"
    assert r.get_json()["data"]["document_count"] == 0
    print("  ✅ PASS\n")

    print("=" * 60)
    print("TEST 4: Nhân viên chỉ tạo được user khi... KHÔNG, chỉ admin mới tạo được user")
    r = client.post("/api/users/", json={"username": "x", "email": "x@test.local", "password": "p12345"},
                     headers=auth_header(emp1_token))
    assert r.status_code == 403, f"❌ Kỳ vọng 403, nhận {r.status_code}"
    r = client.post("/api/users/", json={"username": "emp3", "email": "emp3@test.local", "password": "p12345"},
                     headers=auth_header(admin_token))
    assert r.status_code == 201, f"❌ Admin tạo user thất bại: {r.status_code} {r.get_json()}"
    print("  ✅ PASS\n")

    print("=" * 60)
    print("TEST 5: /register công khai đã bị đóng (404 vì route không còn tồn tại)")
    r = client.post("/api/auth/register", json={"username": "y", "email": "y@test.local", "password": "p12345"})
    assert r.status_code == 404, f"❌ Kỳ vọng 404 (route đã xoá), nhận {r.status_code}"
    print("  ✅ PASS\n")

    print("=" * 60)
    print("TEST 6: Quyền xóa tài liệu — nhân viên chỉ xóa được tài liệu của mình")
    from app.vectorstore.operations import VectorStoreOps
    vector_ops = VectorStoreOps()

    doc_emp1 = Document(filename="a.pdf", original_name="a.pdf", file_type="pdf",
                         file_size_kb=1.0, uploaded_by=emp1_id)
    mongo.db[Document.COLLECTION].insert_one(doc_emp1.to_dict())
    doc_emp1_id = str(doc_emp1._id)

    # emp2 xóa tài liệu của emp1 -> 403
    r = client.delete(f"/api/documents/{doc_emp1_id}", headers=auth_header(emp2_token))
    assert r.status_code == 403, f"❌ Kỳ vọng 403, nhận {r.status_code}: {r.get_json()}"

    # emp1 xóa tài liệu của chính mình -> 200
    r = client.delete(f"/api/documents/{doc_emp1_id}", headers=auth_header(emp1_token))
    assert r.status_code == 200, f"❌ emp1 xóa tài liệu của mình thất bại: {r.status_code} {r.get_json()}"

    # admin xóa tài liệu của người khác -> 200
    doc_emp2 = Document(filename="b.pdf", original_name="b.pdf", file_type="pdf",
                         file_size_kb=1.0, uploaded_by=emp2_id)
    mongo.db[Document.COLLECTION].insert_one(doc_emp2.to_dict())
    r = client.delete(f"/api/documents/{doc_emp2._id}", headers=auth_header(admin_token))
    assert r.status_code == 200, f"❌ Admin xóa tài liệu người khác thất bại: {r.status_code} {r.get_json()}"
    print("  ✅ PASS\n")

    print("=" * 60)
    print("TEST 7: Quyền đọc chat session — nhân viên chỉ đọc được session của mình")
    session_emp1 = ChatSession(user_id=emp1_id, title="test session")
    mongo.db[ChatSession.COLLECTION].insert_one(session_emp1.to_dict())
    session_id = str(session_emp1._id)

    r = client.get(f"/api/chat/sessions/{session_id}", headers=auth_header(emp2_token))
    assert r.status_code == 404, f"❌ Kỳ vọng 404 (bị lọc theo quyền sở hữu), nhận {r.status_code}"

    r = client.get(f"/api/chat/sessions/{session_id}", headers=auth_header(emp1_token))
    assert r.status_code == 200, f"❌ emp1 đọc session của mình thất bại: {r.status_code}"

    r = client.get(f"/api/chat/sessions/{session_id}", headers=auth_header(admin_token))
    assert r.status_code == 200, f"❌ Admin đọc session người khác thất bại: {r.status_code}"
    print("  ✅ PASS\n")

    clean_db()

shutil.rmtree(_tmp_chroma, ignore_errors=True)

print("=" * 60)
print("🎉 TEST AUTH/RBAC HOÀN THÀNH — TẤT CẢ PASS!")
