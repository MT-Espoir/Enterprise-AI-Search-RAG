"""
Test job embed lại header ngữ cảnh (S6): POST/GET /api/admin/context-header/reembed (+ /cancel).
Kiểm tra: chỉ admin; dry_run không ghi gì; embedding = embed("header\\nnội dung") còn documents
giữ nguyên; header lấy theo header HIỆU LỰC (duyệt > gợi ý); chunk đã đúng chuỗi header thì
chỉ đánh dấu, không embed; tài liệu bỏ header → gỡ header khỏi embedding + metadata; tài liệu
chưa ingest xong / không lệch version không bị đụng; khóa chống chạy trùng (409) + job chết
(heartbeat cũ) được chạy lại; hủy giữa chừng rồi chạy lại = làm tiếp phần còn thiếu; header bị
sửa trong lúc chạy → không đánh dấu nhầm. Job chạy khi killswitch TẮT (bật ở S7).

Dùng Flask test client + MongoDB test DB (`rag_test_db`) + ChromaDB tạm + embedder giả 3 chiều.
Chạy: python backend/tests/test_context_header_reembed.py
"""
import sys, os, tempfile, time, hashlib
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_tmp_chroma = tempfile.mkdtemp(prefix="rag_test_chroma_reembed_")
os.environ["CHROMA_PERSIST_PATH"] = _tmp_chroma

from bson import ObjectId

import app.api.admin as admin_api
from app import create_app
from app.core import context_header as ctx
from app.extensions import mongo
from app.models.document import Document
from app.models.user import User
from app.services.auth_service import AuthService
from app.services import context_header_reembed as rj
from app.vectorstore.operations import VectorStoreOps


class FakeEmbedder:
    """Vector 3 chiều xác định theo chuỗi — so được embedding với chuỗi đã embed."""

    def __init__(self, on_batch=None):
        self.texts: list[str] = []
        self.on_batch = on_batch

    @staticmethod
    def vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255 for b in digest[:3]]

    def embed_batch(self, texts):
        self.texts.extend(texts)
        if self.on_batch:
            self.on_batch(len(self.texts))
        return [self.vector(t) for t in texts]


fake = FakeEmbedder()
admin_api._embedder = lambda: fake


def make_user(auth_service, username, email, role="user"):
    password = "Test-pass-123"
    user = auth_service.register(username, email, password)
    if role != "user":
        mongo.db[User.COLLECTION].update_one({"_id": ObjectId(user["_id"])}, {"$set": {"role": role}})
    return auth_service.login(email, password)["access_token"]


def make_doc(filename, *, approved=None, suggested=None, version=0, embedded=0, chunk_header=None,
             n_chunks=3, status=Document.STATUS_DONE):
    doc = Document(filename, filename, "pdf", 10, "owner", status=status,
                   context_header=approved, context_header_suggested=suggested,
                   context_header_version=version, embedded_header_version=embedded)
    mongo.db[Document.COLLECTION].insert_one(doc.to_dict())
    doc_id = str(doc._id)
    VectorStoreOps().add_chunks([{
        "id": f"{doc_id}_p1_c{i}", "text": f"Nội dung {filename} đoạn {i}",
        "embedding": FakeEmbedder.vector(ctx.build_context_text(chunk_header, f"Nội dung {filename} đoạn {i}")),
        "metadata": {"doc_id": doc_id, "filename": filename, "page_num": 1, "chunk_index": i,
                     **({"context_header": chunk_header} if chunk_header else {})},
    } for i in range(n_chunks)])
    return doc_id


def chunks(doc_id):
    return VectorStoreOps().collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas", "embeddings"])


def mongo_doc(doc_id):
    return mongo.db[Document.COLLECTION].find_one({"_id": ObjectId(doc_id)})


def wait_job(client, token, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get("/api/admin/context-header/reembed", headers=auth(token)).get_json()["data"]["job"]
        if job and job["status"] != "running":
            return job
        time.sleep(0.2)
    raise AssertionError("Job không kết thúc trong thời gian chờ")


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def reset():
    for name in (User.COLLECTION, Document.COLLECTION, rj.JOBS_COLLECTION):
        mongo.db[name].delete_many({})
    col = VectorStoreOps().collection
    ids = col.get()["ids"]
    if ids:
        col.delete(ids=ids)


URL = "/api/admin/context-header/reembed"
H_A, H_B, H_C, H_OLD = "[Header A]", "[Header B gợi ý]", "[Header C]", "[Header cũ]"

app = create_app("testing")
client = app.test_client()
assert not ctx.is_enabled(), "Test chạy với killswitch TẮT — job vẫn phải ghép header"

with app.app_context():
    assert mongo.db.name == "rag_test_db", mongo.db.name
    reset()
    auth_service = AuthService()
    admin = make_user(auth_service, "admin", "admin@test.local", role="admin")
    user = make_user(auth_service, "user", "user@test.local")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 1: chunks_to_update — so chuỗi header từng chunk")
    metas = [{"context_header": H_A}, {}, {"context_header": " [Header A] "}, {"context_header": H_OLD}, None]
    assert rj.chunks_to_update(H_A, metas) == [1, 3, 4]
    assert rj.chunks_to_update(None, metas) == [0, 2, 3]
    assert rj.chunks_to_update("", [{}, {"context_header": ""}]) == []
    print("  ✅ PASS\n")

    doc_a = make_doc("a.pdf", approved=H_A, suggested="[gợi ý khác]", version=1)          # duyệt > gợi ý
    doc_b = make_doc("b.pdf", suggested=H_B, version=1)                                     # D1-B: dùng gợi ý
    doc_c = make_doc("c.pdf", approved=H_C, version=2, embedded=1, chunk_header=H_C)        # cùng chuỗi
    doc_d = make_doc("d.pdf", version=3, embedded=2, chunk_header=H_OLD)                    # bỏ header
    doc_e = make_doc("e.pdf", approved="[đang xử lý]", version=1, status=Document.STATUS_PROCESSING)
    doc_f = make_doc("f.pdf", approved=H_A, version=1, embedded=1, chunk_header=H_A)        # không lệch
    original_text = {d: chunks(d)["documents"] for d in (doc_a, doc_b, doc_d)}

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 2: Quyền — chỉ admin")
    assert client.get(URL).status_code == 401
    assert client.get(URL, headers=auth(user)).status_code == 403
    assert client.post(URL, headers=auth(user), json={"dry_run": True}).status_code == 403
    assert client.post(f"{URL}/cancel", headers=auth(user)).status_code == 403
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 3: dry_run / preview — kế hoạch đúng, không ghi gì")
    r = client.post(URL, headers=auth(admin), json={"dry_run": True})
    assert r.status_code == 200, r.get_json()
    plan = r.get_json()["data"]
    by_id = {it["doc_id"]: it for it in plan["items"]}
    assert set(by_id) == {doc_a, doc_b, doc_c, doc_d}, by_id.keys()
    assert (by_id[doc_a]["header"], by_id[doc_b]["header"], by_id[doc_d]["header"]) == (H_A, H_B, None)
    assert by_id[doc_c]["chunks_to_embed"] == 0
    assert (plan["documents"], plan["chunks_to_embed"], plan["documents_mark_only"]) == (4, 9, 1), plan
    preview = client.get(f"{URL}?preview=1", headers=auth(admin)).get_json()["data"]
    assert preview["job"] is None and preview["pending"]["chunks_to_embed"] == 9
    assert fake.texts == [] and mongo_doc(doc_a)["embedded_header_version"] == 0
    assert client.post(URL, headers=auth(admin), json={"doc_ids": "x"}).status_code == 400
    assert client.post(URL, headers=auth(admin), json={"batch_size": 0}).status_code == 400
    assert client.post(URL, headers=auth(admin), json={"doc_ids": ["khong-hop-le"], "dry_run": True}).status_code == 400
    assert client.post(f"{URL}/cancel", headers=auth(admin)).status_code == 409
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 4: Chạy job — embedding có header, documents giữ nguyên, version khớp")
    r = client.post(URL, headers=auth(admin), json={"batch_size": 2})
    assert r.status_code == 202, r.get_json()
    job = wait_job(client, admin)
    assert job["status"] == "done", job
    assert (job["total_docs"], job["processed_docs"], job["embedded_docs"], job["mark_only_docs"]) == (4, 4, 3, 1), job
    assert job["embedded_chunks"] == 9 and job["failed_docs"] == [] and job["bm25_refreshed"] is True, job
    for doc_id, header in ((doc_a, H_A), (doc_b, H_B), (doc_d, None)):
        got = chunks(doc_id)
        assert got["documents"] == original_text[doc_id], "documents phải giữ nội dung gốc"
        for text, meta, emb in zip(got["documents"], got["metadatas"], got["embeddings"]):
            assert meta.get("context_header") == header, (doc_id, meta)
            assert [round(x, 6) for x in emb] == [round(x, 6) for x in FakeEmbedder.vector(ctx.build_context_text(header, text))]
        d = mongo_doc(doc_id)
        assert d["embedded_header_version"] == d["context_header_version"], d
    assert len(fake.texts) == 9 and all(t.startswith(H_A + "\n") for t in fake.texts[:3])
    assert mongo_doc(doc_c)["embedded_header_version"] == 2, "cùng chuỗi → chỉ đánh dấu"
    assert mongo_doc(doc_e)["embedded_header_version"] == 0 and "context_header" not in chunks(doc_e)["metadatas"][0]
    assert mongo_doc(doc_f)["embedded_header_version"] == 1
    # Chạy lại khi không còn gì lệch → 0 tài liệu, không embed
    fake.texts.clear()
    assert client.post(URL, headers=auth(admin), json={}).status_code == 202
    job = wait_job(client, admin)
    assert (job["status"], job["total_docs"], job["embedded_chunks"], fake.texts) == ("done", 0, 0, []), job
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 5: Khóa chống chạy trùng (409) + job chết được chạy lại")
    store = rj.ReembedJobStore()
    assert store.acquire({"total_docs": 0})
    assert client.post(URL, headers=auth(admin), json={}).status_code == 409
    assert not store.acquire({"total_docs": 0})
    stale = datetime.now(timezone.utc) - rj.STALE_AFTER - timedelta(minutes=1)
    mongo.db[rj.JOBS_COLLECTION].update_one({"_id": rj.JOB_ID}, {"$set": {"heartbeat_at": stale}})
    assert client.get(URL, headers=auth(admin)).get_json()["data"]["job"]["status"] == "interrupted"
    assert client.post(URL, headers=auth(admin), json={}).status_code == 202
    assert wait_job(client, admin)["status"] == "done"
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 6: Hủy giữa chừng → chạy lại chỉ làm phần còn thiếu")
    doc_g = make_doc("g.pdf", approved="[Header G]", version=1, n_chunks=3)
    fake.texts.clear()
    fake.on_batch = lambda n: rj.ReembedJobStore().request_cancel() if n == 1 else None
    assert client.post(URL, headers=auth(admin), json={"batch_size": 1}).status_code == 202
    job = wait_job(client, admin)
    assert (job["status"], job["embedded_chunks"], job["processed_docs"]) == ("cancelled", 1, 0), job
    assert mongo_doc(doc_g)["embedded_header_version"] == 0, "hủy giữa tài liệu → chưa đánh dấu"
    assert sum(m.get("context_header") == "[Header G]" for m in chunks(doc_g)["metadatas"]) == 1
    fake.on_batch = None
    fake.texts.clear()
    assert client.post(URL, headers=auth(admin), json={"batch_size": 1}).status_code == 202
    job = wait_job(client, admin)
    assert (job["status"], job["embedded_chunks"], len(fake.texts)) == ("done", 2, 2), job
    assert mongo_doc(doc_g)["embedded_header_version"] == 1
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 7: Header bị sửa trong lúc chạy → không đánh dấu nhầm version")
    doc_h = make_doc("h.pdf", approved="[Header H v1]", version=1, n_chunks=1)
    snapshot = mongo_doc(doc_h)  # job đọc bản này…
    mongo.db[Document.COLLECTION].update_one({"_id": ObjectId(doc_h)},  # …rồi người dùng sửa header
                                            {"$set": {"context_header": "[Header H v2]", "context_header_version": 2}})
    reembedder = rj.ContextHeaderReembedder(VectorStoreOps(), FakeEmbedder())
    count, cancelled = reembedder._reembed_document(snapshot, time.perf_counter(), 0)
    assert (count, cancelled) == (1, False)
    assert mongo_doc(doc_h)["embedded_header_version"] == 0, "version đã đổi → phải còn lệch để lần sau embed lại"
    assert rj.chunks_to_update("[Header H v2]", chunks(doc_h)["metadatas"]) == [0]
    print("  ✅ PASS\n")

    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 8: Chỉ định doc_ids — chỉ xử lý tài liệu được chọn")
    doc_i = make_doc("i.pdf", approved="[Header I]", version=1, n_chunks=2)
    fake.texts.clear()
    assert client.post(URL, headers=auth(admin), json={"doc_ids": [doc_i]}).status_code == 202
    job = wait_job(client, admin)
    assert (job["status"], job["total_docs"], job["embedded_chunks"]) == ("done", 1, 2), job
    assert mongo_doc(doc_h)["embedded_header_version"] == 0, "doc_h không được chọn"
    print("  ✅ PASS\n")

    reset()

print("🎉 TẤT CẢ TEST EMBED LẠI HEADER PASS!")
