"""
Test Document-level ACL theo phòng ban (Department) — xem
roadmap_tasklist/production_target_architecture.md mục 7, plan gốc trong
.claude/plans. Dùng Flask test client + MongoDB test DB (`rag_test_db`) +
ChromaDB tạm, cùng harness với test_auth_rbac.py.

Chạy: python backend/tests/test_document_acl.py
"""
import sys, os, tempfile, shutil

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_tmp_chroma = tempfile.mkdtemp(prefix="rag_test_chroma_acl_")
os.environ["CHROMA_PERSIST_PATH"] = _tmp_chroma

from bson import ObjectId

from app import create_app
from app.extensions import mongo
from app.services.auth_service import AuthService
from app.services.document_service import DocumentService
from app.models.user import User
from app.models.document import Document
from app.vectorstore.operations import VectorStoreOps, build_acl_where, combine_where
from app.core.bm25_index import BM25Index


def clean_db():
    mongo.db[User.COLLECTION].delete_many({})
    mongo.db[Document.COLLECTION].delete_many({})


def make_user(auth_service, username, email, password, role="user", department=""):
    user = auth_service.register(username, email, password)
    update = {}
    if role != "user":
        update["role"] = role
    if department:
        update["department"] = department
    if update:
        mongo.db[User.COLLECTION].update_one({"_id": ObjectId(user["_id"])}, {"$set": update})
    login = auth_service.login(email, password)
    return user["_id"], login["access_token"]


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}


app = create_app("testing")
client = app.test_client()

with app.app_context():
    clean_db()

    # ══════════════════════════════════════════════════════════════
    # TEST 1: build_acl_where() / combine_where() — pure unit
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 1: build_acl_where() / combine_where()")

    assert build_acl_where("") == {"department": ""}
    assert build_acl_where(None) == {"department": ""}
    assert build_acl_where("hr") == {"$or": [{"department": ""}, {"department": "hr"}]}

    assert combine_where(None, None) is None
    assert combine_where({"a": 1}, None) == {"a": 1}
    assert combine_where(None, {"b": 2}) == {"b": 2}
    assert combine_where({"a": 1}, {"b": 2}) == {"$and": [{"a": 1}, {"b": 2}]}
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    # TEST 2: BM25Index.search() ACL — OR semantics
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 2: BM25Index.search() Document-level ACL")

    bm25 = BM25Index()
    bm25.build([
        {"id": "c1", "text": "quy định chung áp dụng mọi phòng ban", "metadata": {"doc_id": "d1", "department": ""}},
        {"id": "c2", "text": "quy định nội bộ phòng nhân sự", "metadata": {"doc_id": "d2", "department": "hr"}},
        {"id": "c3", "text": "quy định nội bộ phòng công nghệ thông tin", "metadata": {"doc_id": "d3", "department": "it"}},
    ])

    r = bm25.search("quy định", top_k=10, acl_bypass=True)
    assert len(r) == 3, f"❌ acl_bypass=True phải thấy cả 3: {r}"

    r = bm25.search("quy định", top_k=10, acl_department="hr", acl_bypass=False)
    ids = {x["id"] for x in r}
    assert ids == {"c1", "c2"}, f"❌ department=hr phải thấy c1(mở)+c2(hr), không thấy c3(it): {ids}"

    r = bm25.search("quy định", top_k=10, acl_department="", acl_bypass=False)
    ids = {x["id"] for x in r}
    assert ids == {"c1"}, f"❌ department rỗng chỉ được thấy c1 (mở): {ids}"

    r = bm25.search("quy định", top_k=10, acl_department="finance", acl_bypass=False)
    ids = {x["id"] for x in r}
    assert ids == {"c1"}, f"❌ department=finance (không khớp chunk nào) chỉ thấy c1: {ids}"
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    # TEST 3: VectorStoreOps.get_first_chunks_of_doc(acl_where=...) — Chroma thật (tạm)
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 3: get_first_chunks_of_doc() với acl_where")

    vector_ops = VectorStoreOps()
    dummy_embedding = [0.1] * 8
    vector_ops.add_chunks([
        {"id": "acl_c1", "text": "chunk mở", "embedding": dummy_embedding,
         "metadata": {"doc_id": "acl_doc", "department": "", "chunk_index": 0, "filename": "x.pdf"}},
        {"id": "acl_c2", "text": "chunk hr", "embedding": dummy_embedding,
         "metadata": {"doc_id": "acl_doc", "department": "hr", "chunk_index": 1, "filename": "x.pdf"}},
    ])

    r = vector_ops.get_first_chunks_of_doc("acl_doc", limit=10, acl_where=None)
    assert len(r) == 2, f"❌ Không ACL phải thấy cả 2 chunk: {r}"

    r = vector_ops.get_first_chunks_of_doc("acl_doc", limit=10, acl_where=build_acl_where("hr"))
    assert len(r) == 2, f"❌ department=hr phải thấy cả 2 (mở + hr): {r}"

    r = vector_ops.get_first_chunks_of_doc("acl_doc", limit=10, acl_where=build_acl_where("it"))
    assert len(r) == 1 and r[0]["id"] == "acl_c1", f"❌ department=it chỉ được thấy chunk mở: {r}"
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    # TEST 4: DocumentService.get_documents()/get_document_by_id() — Mongo ACL
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 4: get_documents()/get_document_by_id() Mongo ACL")

    doc_service = DocumentService(vector_ops, upload_folder="/tmp")

    doc_open = Document(filename="open.pdf", original_name="open.pdf", file_type="pdf", file_size_kb=1.0, uploaded_by="u1")
    doc_hr = Document(filename="hr.pdf", original_name="hr.pdf", file_type="pdf", file_size_kb=1.0, uploaded_by="u1", department="hr")
    doc_it = Document(filename="it.pdf", original_name="it.pdf", file_type="pdf", file_size_kb=1.0, uploaded_by="u1", department="it")
    for d in (doc_open, doc_hr, doc_it):
        mongo.db[Document.COLLECTION].insert_one(d.to_dict())

    result = doc_service.get_documents(acl_department="hr", acl_bypass=False)
    names = {d["original_name"] for d in result["documents"]}
    assert names == {"open.pdf", "hr.pdf"}, f"❌ user hr phải thấy open+hr, không thấy it: {names}"

    result = doc_service.get_documents(acl_bypass=True)
    names = {d["original_name"] for d in result["documents"]}
    assert names == {"open.pdf", "hr.pdf", "it.pdf"}, f"❌ admin (bypass) phải thấy cả 3: {names}"

    doc = doc_service.get_document_by_id(str(doc_it._id), acl_department="hr", acl_bypass=False)
    assert doc is None, f"❌ user hr KHÔNG được lấy doc it: {doc}"

    doc = doc_service.get_document_by_id(str(doc_hr._id), acl_department="hr", acl_bypass=False)
    assert doc is not None, "❌ user hr phải lấy được doc hr"

    doc = doc_service.get_document_by_id(str(doc_open._id), acl_department="hr", acl_bypass=False)
    assert doc is not None, "❌ user hr phải lấy được doc mở (chưa gán)"

    doc = doc_service.get_document_by_id(str(doc_it._id), acl_bypass=True)
    assert doc is not None, "❌ admin (bypass) phải lấy được doc it"

    # Mặc định acl_bypass=True (không đổi hành vi caller nội bộ delete/update)
    doc = doc_service.get_document_by_id(str(doc_it._id))
    assert doc is not None, "❌ Mặc định acl_bypass=True phải lấy được (backward-compat delete/update)"
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    # TEST 5: End-to-end qua Flask test client — 2 user khác phòng ban
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 5: End-to-end — emp_hr / emp_it / admin qua API thật")

    auth_service = AuthService()
    admin_id, admin_token = make_user(auth_service, "admin_acl", "admin_acl@test.local", "pass1234", role="admin")
    hr_id, hr_token = make_user(auth_service, "emp_hr", "emp_hr@test.local", "pass1234", department="hr")
    it_id, it_token = make_user(auth_service, "emp_it", "emp_it@test.local", "pass1234", department="it")

    r = client.get("/api/documents/", headers=auth_header(hr_token))
    assert r.status_code == 200
    hr_names = {d["original_name"] for d in r.get_json()["documents"]}
    assert "it.pdf" not in hr_names, f"❌ emp_hr KHÔNG được thấy it.pdf trong list: {hr_names}"
    assert "hr.pdf" in hr_names and "open.pdf" in hr_names, f"❌ emp_hr phải thấy hr.pdf+open.pdf: {hr_names}"

    r = client.get(f"/api/documents/{doc_it._id}", headers=auth_header(hr_token))
    assert r.status_code == 404, f"❌ emp_hr fetch doc it phải 404 (không xác nhận tồn tại), nhận {r.status_code}"

    r = client.get(f"/api/documents/{doc_it._id}", headers=auth_header(it_token))
    assert r.status_code == 200, f"❌ emp_it fetch đúng doc it phải 200, nhận {r.status_code}"

    r = client.get(f"/api/documents/{doc_it._id}", headers=auth_header(admin_token))
    assert r.status_code == 200, f"❌ admin fetch bất kỳ doc nào phải 200, nhận {r.status_code}"

    # Chỉ admin được đổi department
    r = client.patch(f"/api/documents/{doc_open._id}/metadata", json={"department": "hr"}, headers=auth_header(hr_token))
    assert r.status_code == 403, f"❌ Non-admin đổi department phải 403, nhận {r.status_code}: {r.get_json()}"

    r = client.patch(f"/api/documents/{doc_open._id}/metadata", json={"department": "hr"}, headers=auth_header(admin_token))
    assert r.status_code == 200, f"❌ Admin đổi department phải 200, nhận {r.status_code}: {r.get_json()}"
    # Trả lại "" để không ảnh hưởng test khác nếu re-run
    client.patch(f"/api/documents/{doc_open._id}/metadata", json={"department": ""}, headers=auth_header(admin_token))
    print("  ✅ PASS\n")

    clean_db()

shutil.rmtree(_tmp_chroma, ignore_errors=True)

print("=" * 60)
print("🎉 TEST DOCUMENT ACL HOÀN THÀNH — TẤT CẢ PASS!")
