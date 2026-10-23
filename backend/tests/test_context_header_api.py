"""
Test API header ngữ cảnh: GET / PATCH /api/documents/<id>/context-header và
POST /api/documents/<id>/context-header/suggest — luồng gợi ý → duyệt → sửa → bỏ duyệt,
quyền (chủ sở hữu / admin / người khác), ACL đọc theo phòng ban (404 không phải 403),
body không hợp lệ. Dùng Flask test client + MongoDB test DB (`rag_test_db`) + ChromaDB tạm,
cùng harness với test_document_acl.py. SLM bị chặn: rule phải đủ cho văn bản mẫu.

Chạy: python backend/tests/test_context_header_api.py
"""
import sys, os, tempfile

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_tmp_chroma = tempfile.mkdtemp(prefix="rag_test_chroma_header_")
os.environ["CHROMA_PERSIST_PATH"] = _tmp_chroma

from bson import ObjectId

import app.core.context_header_suggest as sg
from app import create_app
from app.extensions import mongo
from app.services.auth_service import AuthService
from app.models.user import User
from app.models.document import Document
from app.models.department import DEPARTMENTS
from app.vectorstore.operations import VectorStoreOps


def _refuse_slm(*a, **kw):
    raise AssertionError("Văn bản mẫu đủ cho rule — không được gọi SLM")


sg.requests.post = _refuse_slm

DEPT_A, DEPT_B = DEPARTMENTS[0], DEPARTMENTS[1]
DECREE_TEXT = ("CHÍNH PHỦ CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Số: 145/2020/NĐ-CP NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT "
               "MỘT SỐ ĐIỀU CỦA BỘ LUẬT LAO ĐỘNG Căn cứ Bộ luật Lao động ngày 20 tháng 11 năm 2019;")
SUGGESTED = "[Nghị định số 145/2020/NĐ-CP quy định chi tiết một số điều của bộ luật lao động]"


def make_user(auth_service, username, email, role="user", department=""):
    password = "Test-pass-123"
    user = auth_service.register(username, email, password)
    update = {k: v for k, v in (("role", role), ("department", department)) if v and v != "user"}
    if update:
        mongo.db[User.COLLECTION].update_one({"_id": ObjectId(user["_id"])}, {"$set": update})
    return user["_id"], auth_service.login(email, password)["access_token"]


def make_doc(owner_id, filename, department="", with_chunks=True):
    doc = Document(filename, filename, "pdf", 10, owner_id, status=Document.STATUS_DONE, department=department)
    mongo.db[Document.COLLECTION].insert_one(doc.to_dict())
    doc_id = str(doc._id)
    if with_chunks:
        VectorStoreOps().add_chunks([{
            "id": f"{doc_id}_p1_c0", "text": DECREE_TEXT, "embedding": [0.1, 0.2, 0.3],
            "metadata": {"doc_id": doc_id, "filename": filename, "page_num": 1, "chunk_index": 0, "department": department},
        }])
    return doc_id


def auth(token):
    return {"Authorization": f"Bearer {token}"}


app = create_app("testing")
client = app.test_client()

with app.app_context():
    mongo.db[User.COLLECTION].delete_many({})
    mongo.db[Document.COLLECTION].delete_many({})
    auth_service = AuthService()
    owner_id, owner = make_user(auth_service, "owner", "owner@test.local", department=DEPT_A)
    _, other = make_user(auth_service, "other", "other@test.local", department=DEPT_A)
    _, outsider = make_user(auth_service, "outsider", "outsider@test.local", department=DEPT_B)
    _, admin = make_user(auth_service, "admin", "admin@test.local", role="admin")

    doc_id = make_doc(owner_id, "1452020nd-cp.pdf")
    url = f"/api/documents/{doc_id}/context-header"

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 1: Tài liệu cũ chưa có header")
    r = client.get(url, headers=auth(owner))
    assert r.status_code == 200, r.get_json()
    s = r.get_json()
    assert (s["status"], s["effective"], s["version"], s["needs_reembed"]) == ("none", None, 0, False), s
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 2: POST suggest — chỉ chủ sở hữu/admin; gợi ý rule từ chunk đã lưu")
    assert client.post(url + "/suggest", headers=auth(other)).status_code == 403
    r = client.post(url + "/suggest", headers=auth(owner))
    assert r.status_code == 200, r.get_json()
    s = r.get_json()
    assert s["status"] == "suggested" and s["suggested"] == SUGGESTED and s["suggested_source"] == "rule", s
    assert (s["version"], s["embedded_version"], s["needs_reembed"]) == (1, 0, True), s
    stored = mongo.db[Document.COLLECTION].find_one({"_id": ObjectId(doc_id)})
    assert stored["context_header_suggested"] == SUGGESTED and stored["context_header_version"] == 1
    print(f"  {s['suggested']}\n  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 3: PATCH approve / sửa / bỏ duyệt — version theo header hiệu lực")
    r = client.patch(url, json={"approve": True}, headers=auth(owner))
    s = r.get_json()
    assert r.status_code == 200 and s["status"] == "approved" and s["version"] == 1, s
    assert s["source"] == "rule", "duyệt nguyên văn → giữ nguồn rule, KHÔNG tăng version"

    custom = "  Nghị định 145/2020/NĐ-CP về điều kiện lao động  "
    s = client.patch(url, json={"context_header": custom}, headers=auth(admin)).get_json()
    assert s["approved"] == "[Nghị định 145/2020/NĐ-CP về điều kiện lao động]", s
    assert (s["source"], s["version"]) == ("user", 2), s

    s = client.patch(url, json={"context_header": None}, headers=auth(owner)).get_json()
    assert (s["status"], s["effective"], s["version"]) == ("suggested", SUGGESTED, 3), s

    r = client.post(url + "/suggest", headers=auth(owner))
    assert r.get_json()["version"] == 3, "gợi ý lại ra cùng chuỗi → không đổi header hiệu lực"
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 4: Quyền + body không hợp lệ")
    assert client.patch(url, json={"approve": True}, headers=auth(other)).status_code == 403
    assert client.patch(url, json={}, headers=auth(owner)).status_code == 400
    assert client.patch(url, json={"context_header": 123}, headers=auth(owner)).status_code == 400
    r = client.patch(url, json={"context_header": "x" * 300}, headers=auth(owner))
    assert r.status_code == 400 and "tối đa" in r.get_json()["error"], r.get_json()
    assert client.get(url).status_code == 401
    missing = f"/api/documents/{ObjectId()}/context-header"
    assert client.get(missing, headers=auth(owner)).status_code == 404
    assert client.patch(missing, json={"approve": True}, headers=auth(admin)).status_code == 404

    bare = make_doc(owner_id, "no-chunks.pdf", with_chunks=False)
    r = client.patch(f"/api/documents/{bare}/context-header", json={"approve": True}, headers=auth(owner))
    assert r.status_code == 400, "chưa có gợi ý thì không duyệt được"
    r = client.post(f"/api/documents/{bare}/context-header/suggest", headers=auth(owner))
    assert r.status_code == 400, "chưa có chunk thì không gợi ý được"
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 5: ACL đọc — khác phòng ban → 404 (không xác nhận tồn tại), admin thấy")
    secret = make_doc(owner_id, "secret.pdf", department=DEPT_A)
    secret_url = f"/api/documents/{secret}/context-header"
    assert client.get(secret_url, headers=auth(outsider)).status_code == 404
    assert client.get(secret_url, headers=auth(other)).status_code == 200
    assert client.get(secret_url, headers=auth(admin)).status_code == 200
    print("  ✅ PASS\n")

    mongo.db[User.COLLECTION].delete_many({})
    mongo.db[Document.COLLECTION].delete_many({})

print("🎉 TẤT CẢ TEST API HEADER PASS!")
