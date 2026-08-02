"""
Test 3 gate Phase 5 Production Hardening (Layer 6):
  Gate 1 — Rate limiting (Flask-Limiter)
  Gate 2 — Multi-worker BM25 sync (mark_bm25_dirty fail-open + version marker)
  Gate 3 — Upload dedup (RAG-011, SHA-256)

Chạy trực tiếp (KHÔNG qua pytest, theo convention test khác của dự án):
  PYTHONIOENCODING=utf-8 ml_env/Scripts/python.exe tests/test_production_hardening.py

Yêu cầu: MongoDB chạy ở localhost:27017 (chỉ Gate 3 cần — sẽ SKIP nếu không có).
KHÔNG cần Redis (test đường fail-open + dùng fake redis).
"""
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from werkzeug.datastructures import FileStorage

from app import create_app
from app.extensions import limiter


# ══════════════════════════════════════════════════════════════════
# GATE 1 — RATE LIMITING
# ══════════════════════════════════════════════════════════════════
def test_rate_limiting():
    print("=" * 60)
    print("GATE 1 — Rate limiting")
    app = create_app("testing")
    # Ép giới hạn nhỏ + rõ ràng để test nhanh, không phụ thuộc default.
    app.config["RATELIMIT_LOGIN"] = "3 per minute"
    client = app.test_client()

    # 3 request đầu (body rỗng -> route trả 400 TRƯỚC khi chạm Mongo) KHÔNG bị
    # chặn; request thứ 4 vượt giới hạn -> 429. Limiter đếm mọi hit bất kể status.
    statuses = []
    for _ in range(4):
        r = client.post("/api/auth/login", json={}, environ_base={"REMOTE_ADDR": "10.0.0.1"})
        statuses.append(r.status_code)
    print(f"  4 request cùng IP: statuses = {statuses}")
    assert statuses[:3] == [400, 400, 400], f"3 request đầu không được bị chặn: {statuses}"
    assert statuses[3] == 429, f"Request thứ 4 phải bị 429, got {statuses[3]}"

    body = client.post("/api/auth/login", json={}, environ_base={"REMOTE_ADDR": "10.0.0.1"}).get_json()
    assert body["error"]["code"] == "rate_limit_exceeded", f"429 body sai format: {body}"
    print(f"  429 body format đúng: code={body['error']['code']!r}")

    # IP KHÁC không bị ảnh hưởng (key theo IP) — chứng minh không chặn nhầm user khác.
    r_other = client.post("/api/auth/login", json={}, environ_base={"REMOTE_ADDR": "10.0.0.99"})
    assert r_other.status_code == 400, f"IP khác phải không bị chặn, got {r_other.status_code}"
    print(f"  IP khác (10.0.0.99) KHÔNG bị chặn: {r_other.status_code}")
    print("  ✅ PASS: rate-limit chặn đúng theo IP, 429 đúng format, không chặn nhầm IP khác.\n")


def test_rate_limit_killswitch():
    print("=" * 60)
    print("GATE 1b — Rate limiting killswitch (RATELIMIT_ENABLED=false)")
    app = create_app("testing")
    app.config["RATELIMIT_LOGIN"] = "1 per minute"
    # Killswitch: limiter.enabled do Flask-Limiter đọc từ config RATELIMIT_ENABLED
    # lúc init; ở đây set trực tiếp trên singleton để mô phỏng tắt.
    limiter.enabled = False
    try:
        client = app.test_client()
        statuses = [client.post("/api/auth/login", json={},
                                environ_base={"REMOTE_ADDR": "10.0.1.1"}).status_code for _ in range(5)]
        print(f"  5 request khi tắt rate-limit: {statuses}")
        assert 429 not in statuses, f"Killswitch tắt mà vẫn bị 429: {statuses}"
        print("  ✅ PASS: tắt RATELIMIT_ENABLED thì không còn 429.\n")
    finally:
        limiter.enabled = True  # khôi phục cho test khác


# ══════════════════════════════════════════════════════════════════
# GATE 2 — MULTI-WORKER BM25 SYNC
# ══════════════════════════════════════════════════════════════════
def test_bm25_sync_marker():
    print("=" * 60)
    print("GATE 2 — BM25 cross-worker sync marker")
    from app.core.bm25_sync import mark_bm25_dirty

    # (a) Fail-open: không có Redis -> KHÔNG raise.
    mark_bm25_dirty(None)
    print("  mark_bm25_dirty(None) không raise (fail-open) ✅")

    # (b) Có Redis (fake): gọi INCR đúng key.
    class FakeRedis:
        def __init__(self): self.store = {}
        def incr(self, key): self.store[key] = self.store.get(key, 0) + 1; return self.store[key]
        def get(self, key): return self.store.get(key)

    fake = FakeRedis()
    mark_bm25_dirty(fake)
    mark_bm25_dirty(fake)
    assert fake.store.get("bm25:version") == 2, f"marker phải = 2, got {fake.store}"
    print(f"  mark_bm25_dirty(fake_redis) x2 -> bm25:version = {fake.store['bm25:version']} ✅")

    # (c) Redis lỗi lúc incr -> vẫn fail-open (không raise).
    class BrokenRedis:
        def incr(self, key): raise RuntimeError("redis down")
    mark_bm25_dirty(BrokenRedis())
    print("  mark_bm25_dirty(broken_redis) nuốt lỗi, không raise (fail-open) ✅")
    print("  ✅ PASS: marker tăng đúng khi có Redis, fail-open khi không/lỗi Redis.\n")


def test_bm25_sync_worker_initial_build():
    print("=" * 60)
    print("GATE 2b — BM25 sync worker build index lần đầu lúc khởi động")
    import app.core.bm25_sync as bm25_sync

    app = create_app("testing")
    built = {"called": False}

    # Giả lập get_bm25_index().refresh() để không phụ thuộc Chroma thật.
    class FakeIndex:
        def refresh(self, ops): built["called"] = True; return True
    # Patch trong module bm25_sync (import cục bộ bên trong _build_index_once).
    import app.core.bm25_index as bm25_index_mod
    orig = bm25_index_mod.get_bm25_index
    bm25_index_mod.get_bm25_index = lambda: FakeIndex()
    # VectorStoreOps() được gọi trong app context — patch để không chạm Chroma.
    import app.vectorstore.operations as ops_mod
    orig_ops = ops_mod.VectorStoreOps
    ops_mod.VectorStoreOps = lambda: object()
    try:
        t = bm25_sync.start_bm25_sync_worker(app, redis_client=None, interval=1)
        t.join(timeout=5)  # không Redis -> build 1 lần rồi thread thoát
        assert built["called"], "Worker phải gọi refresh() để build index lần đầu"
        assert not t.is_alive(), "Không có Redis thì thread phải thoát sau khi build 1 lần"
        print("  Worker build index lần đầu + thoát khi không có Redis ✅")
    finally:
        bm25_index_mod.get_bm25_index = orig
        ops_mod.VectorStoreOps = orig_ops
    print("  ✅ PASS: build-on-startup hoạt động (xử lý bug BM25 rỗng sau restart).\n")


# ══════════════════════════════════════════════════════════════════
# GATE 3 — UPLOAD DEDUP (RAG-011)
# ══════════════════════════════════════════════════════════════════
def test_file_hash():
    print("=" * 60)
    print("GATE 3a — _compute_file_hash (SHA-256, seek về 0 sau khi băm)")
    from app.services.document_service import _compute_file_hash
    import hashlib

    content = b"noi dung file test 12345"
    fs = FileStorage(stream=io.BytesIO(content), filename="a.txt")
    h = _compute_file_hash(fs)
    assert h == hashlib.sha256(content).hexdigest(), "hash sai"
    # Sau khi băm, stream phải seek về 0 để save_file() đọc lại được đầy đủ.
    assert fs.stream.read() == content, "stream không được seek về 0 sau khi băm"
    print(f"  hash đúng + stream seek về 0 sau băm ✅")
    print("  ✅ PASS\n")


def test_upload_dedup():
    print("=" * 60)
    print("GATE 3b — Upload dedup (cần MongoDB)")
    try:
        from pymongo import MongoClient
        MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1500).admin.command("ping")
    except Exception as e:
        print(f"  ⏭️  SKIP: MongoDB không sẵn sàng ({e})\n")
        return

    from app.services.document_service import DocumentService, DuplicateDocumentError
    from app.extensions import mongo
    from app.models.document import Document

    app = create_app("testing")
    tmp_upload = os.path.join(os.path.dirname(__file__), "_tmp_dedup_upload")
    os.makedirs(tmp_upload, exist_ok=True)

    created_ids = []
    with app.app_context():
        svc = DocumentService(vector_ops=None, upload_folder=tmp_upload)
        content = b"file trung lap dedup test %PDF-fake"

        # Upload lần 1 -> OK
        fs1 = FileStorage(stream=io.BytesIO(content), filename="doc.txt")
        info1 = svc.save_uploaded_file(fs1, user_id="tester")
        created_ids.append(info1["doc_id"])
        print(f"  Upload lần 1: doc_id={info1['doc_id']} ✅")

        # Verify file_hash được lưu
        from bson import ObjectId
        rec = mongo.db[Document.COLLECTION].find_one({"_id": ObjectId(info1["doc_id"])})
        assert rec.get("file_hash"), "file_hash không được lưu vào Mongo"
        print(f"  file_hash đã lưu: {rec['file_hash'][:16]}... ✅")

        # Upload lần 2 CÙNG nội dung -> DuplicateDocumentError, trả doc_id cũ
        fs2 = FileStorage(stream=io.BytesIO(content), filename="doc-copy.txt")
        try:
            svc.save_uploaded_file(fs2, user_id="tester")
            assert False, "Upload trùng phải raise DuplicateDocumentError"
        except DuplicateDocumentError as de:
            assert de.existing_doc_id == info1["doc_id"], "existing_doc_id sai"
            print(f"  Upload lần 2 (trùng) -> DuplicateDocumentError, existing_doc_id={de.existing_doc_id} ✅")

        # Nội dung KHÁC -> OK (không chặn nhầm)
        fs3 = FileStorage(stream=io.BytesIO(content + b" khac"), filename="doc2.txt")
        info3 = svc.save_uploaded_file(fs3, user_id="tester")
        created_ids.append(info3["doc_id"])
        print(f"  Upload nội dung khác -> OK, doc_id={info3['doc_id']} ✅")

        # Dọn dẹp
        for did in created_ids:
            mongo.db[Document.COLLECTION].delete_one({"_id": ObjectId(did)})
    # Xóa file tạm
    for f in os.listdir(tmp_upload):
        try: os.remove(os.path.join(tmp_upload, f))
        except Exception: pass
    try: os.rmdir(tmp_upload)
    except Exception: pass
    print("  ✅ PASS: chặn upload trùng, lưu file_hash, không chặn nhầm file khác.\n")


# ══════════════════════════════════════════════════════════════════
# FIX 🔴#1 — ACL/METADATA STALE TRÊN BM25 (rò rỉ tài liệu)
# ══════════════════════════════════════════════════════════════════
def test_bm25_acl_metadata_rebuild():
    print("=" * 60)
    print("FIX 🔴#1 — BM25 rebuild khi đổi ACL/metadata (chống rò rỉ)")
    from app.core.bm25_index import BM25Index

    idx = BM25Index()
    # v1: chunk MỞ (department="") -> mọi user thấy
    idx.build([{"id": "c1", "text": "bang luong nhan vien thang 1", "metadata": {"department": ""}}])
    assert len(idx.search("luong", acl_department="Phong X")) == 1
    print("  v1 (mở): user 'Phong X' thấy c1 ✅")

    # Đổi ACL: c1 giờ hạn chế "Phong Ke Toan". TRƯỚC FIX (checksum ID-only) ->
    # build() trả False (bỏ qua rebuild) -> BM25 vẫn lọc theo department cũ "".
    rebuilt = idx.build([{"id": "c1", "text": "bang luong nhan vien thang 1", "metadata": {"department": "Phong Ke Toan"}}])
    assert rebuilt is True, "Checksum PHẢI phát hiện đổi department -> rebuild (trước fix: ID-only nên skip -> rò rỉ)"
    print("  đổi department '' -> 'Phong Ke Toan': build() rebuild=True (checksum gồm metadata) ✅")

    # Sau rebuild: user 'Phong X' KHÔNG còn thấy c1 (không rò rỉ)
    leaked = idx.search("luong", acl_department="Phong X")
    assert len(leaked) == 0, f"RÒ RỈ ACL: BM25 vẫn trả chunk đã hạn chế cho user sai phòng: {leaked}"
    print("  user 'Phong X' KHÔNG còn thấy c1 (rò rỉ đã bịt) ✅")
    # user đúng phòng vẫn thấy
    assert len(idx.search("luong", acl_department="Phong Ke Toan")) == 1
    print("  user 'Phong Ke Toan' vẫn thấy c1 (không chặn nhầm) ✅")

    # Không đổi gì -> KHÔNG rebuild (giữ tối ưu checksum, không phá perf)
    assert idx.build([{"id": "c1", "text": "bang luong nhan vien thang 1", "metadata": {"department": "Phong Ke Toan"}}]) is False
    print("  build lại y hệt -> rebuild=False (giữ tối ưu checksum) ✅")
    print("  ✅ PASS: đổi ACL/metadata trigger rebuild, bịt rò rỉ, không phá tối ưu.\n")


# ══════════════════════════════════════════════════════════════════
# FIX 🟠#3 — RATE-LIMIT KEY PER-USER (enterprise chung IP)
# ══════════════════════════════════════════════════════════════════
def test_rate_limit_key_per_user():
    print("=" * 60)
    print("FIX 🟠#3 — user_or_ip_key: per-user khi có JWT, IP khi không")
    from app.extensions import user_or_ip_key
    from flask_jwt_extended import create_access_token

    app = create_app("testing")
    with app.app_context():
        tok_a = create_access_token(identity="userA")
        tok_b = create_access_token(identity="userB")

    # Không token -> fallback IP
    with app.test_request_context("/", environ_base={"REMOTE_ADDR": "9.9.9.9"}):
        k = user_or_ip_key()
        assert k == "9.9.9.9", f"không token phải fallback IP, got {k}"
    print("  không JWT -> key = IP (9.9.9.9) ✅")

    # Có token -> per-user
    with app.test_request_context("/", headers={"Authorization": f"Bearer {tok_a}"}, environ_base={"REMOTE_ADDR": "9.9.9.9"}):
        assert user_or_ip_key() == "user:userA"
    # User khác, CÙNG IP -> key KHÁC (chứng minh không chung bucket theo IP)
    with app.test_request_context("/", headers={"Authorization": f"Bearer {tok_b}"}, environ_base={"REMOTE_ADDR": "9.9.9.9"}):
        assert user_or_ip_key() == "user:userB"
    print("  2 user khác nhau CÙNG IP -> key khác nhau (user:userA vs user:userB) ✅")
    print("  ✅ PASS: rate-limit tách theo user, enterprise chung IP không bị chặn nhầm.\n")


# ══════════════════════════════════════════════════════════════════
# FIX 🟠#4 — CHROMA PERSISTENT MULTI-WORKER FAIL-LOUD
# ══════════════════════════════════════════════════════════════════
def test_chroma_persistent_failloud():
    print("=" * 60)
    print("FIX 🟠#4 — production + CHROMA_MODE=persistent -> RuntimeError (fail loud)")
    import subprocess, sys as _sys, os as _os
    backend = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    base_env = dict(_os.environ,
                    SECRET_KEY="x" * 40, JWT_SECRET_KEY="y" * 40,
                    CHROMA_MODE="persistent", PYTHONIOENCODING="utf-8")

    # (a) persistent + KHÔNG có escape hatch -> phải raise
    code = "from app import create_app\ntry:\n create_app('production'); print('NO_ERROR')\nexcept RuntimeError as e:\n print('RUNTIME_ERROR' if 'persistent' in str(e).lower() else 'OTHER_ERROR')"
    r1 = subprocess.run([_sys.executable, "-c", code], cwd=backend,
                        env=dict(base_env, ALLOW_SINGLE_WORKER_PERSISTENT="false"),
                        capture_output=True, text=True)
    assert "RUNTIME_ERROR" in r1.stdout, f"persistent phải raise RuntimeError: {r1.stdout} {r1.stderr[-300:]}"
    print("  persistent (không escape hatch) -> RuntimeError ✅")

    # (b) persistent + ALLOW_SINGLE_WORKER_PERSISTENT=true -> KHÔNG raise (cố tình 1 worker)
    r2 = subprocess.run([_sys.executable, "-c", code], cwd=backend,
                        env=dict(base_env, ALLOW_SINGLE_WORKER_PERSISTENT="true"),
                        capture_output=True, text=True)
    assert "NO_ERROR" in r2.stdout, f"escape hatch phải cho chạy: {r2.stdout} {r2.stderr[-300:]}"
    print("  persistent + ALLOW_SINGLE_WORKER_PERSISTENT=true -> chạy được (cố tình 1 worker) ✅")
    print("  ✅ PASS: fail-loud đúng, có escape hatch có chủ đích.\n")


if __name__ == "__main__":
    test_rate_limiting()
    test_rate_limit_killswitch()
    test_bm25_sync_marker()
    test_bm25_sync_worker_initial_build()
    test_file_hash()
    test_upload_dedup()
    test_bm25_acl_metadata_rebuild()
    test_rate_limit_key_per_user()
    test_chroma_persistent_failloud()
    print("=" * 60)
    print("🎉 TEST PRODUCTION HARDENING (3 GATE + 4 FIX) HOÀN THÀNH — TẤT CẢ PASS!")
