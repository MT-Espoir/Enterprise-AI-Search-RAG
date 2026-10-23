"""
Test áp HNSW ef_search cho collection (app/vectorstore/client.py, CHROMA_HNSW_EF_SEARCH):
collection mới / collection cũ đang ef_search mặc định 100 → nâng lên giá trị cấu hình ở lần lấy
collection ĐẦU TIÊN của tiến trình (trước mọi truy vấn — ef_search chốt khi index nạp); đã đúng
thì không modify lại; CHROMA_HNSW_EF_SEARCH=0 → giữ nguyên.

Dùng ChromaDB tạm (không đụng chromadb_data production).
Chạy: python backend/tests/test_chroma_ef_search.py
"""
import sys, os, tempfile

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb

import app.vectorstore.client as vc
from app import create_app


def ef_search_of(path, name="enterprise_documents"):
    return chromadb.PersistentClient(path=path).get_collection(name).configuration["hnsw"]["ef_search"]


def fresh_app(path, ef):
    os.environ["CHROMA_PERSIST_PATH"] = path
    app = create_app("testing")
    app.config["CHROMA_PERSIST_PATH"] = path
    app.config["CHROMA_HNSW_EF_SEARCH"] = ef
    vc._client = None
    vc._ef_search_applied = False
    return app


print("=" * 60)
print("TEST 1: Collection mới → ef_search = cấu hình")
path = tempfile.mkdtemp(prefix="rag_test_chroma_ef_")
app = fresh_app(path, 500)
with app.app_context():
    col = vc.get_collection(vc.get_chroma_client())
    assert col.configuration["hnsw"]["ef_search"] == 500, col.configuration
    assert col.configuration["hnsw"]["space"] == "cosine"
print("  ✅ PASS\n")

print("=" * 60)
print("TEST 2: Collection cũ ef_search 100 (mặc định) → nâng lên ở lần lấy đầu tiên, dữ liệu giữ nguyên")
path = tempfile.mkdtemp(prefix="rag_test_chroma_ef_")
old = chromadb.PersistentClient(path=path).get_or_create_collection("enterprise_documents", metadata={"hnsw:space": "cosine"})
old.add(ids=["a", "b"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], documents=["x", "y"])
assert old.configuration["hnsw"]["ef_search"] == 100
del old
app = fresh_app(path, 500)
with app.app_context():
    col = vc.get_collection(vc.get_chroma_client())
    assert col.configuration["hnsw"]["ef_search"] == 500
    assert col.count() == 2 and col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)["ids"][0] == ["a"]
    assert vc._ef_search_applied is True
    # Lần lấy sau trong cùng tiến trình: không modify lại (đặt tay giá trị khác để phát hiện)
    col.modify(configuration={"hnsw": {"ef_search": 321}})
    vc.get_collection(vc.get_chroma_client())
    assert ef_search_of(path) == 321, "chỉ áp 1 lần mỗi tiến trình"
print("  ✅ PASS\n")

print("=" * 60)
print("TEST 3: CHROMA_HNSW_EF_SEARCH = 0 → giữ nguyên ef_search của collection")
path = tempfile.mkdtemp(prefix="rag_test_chroma_ef_")
chromadb.PersistentClient(path=path).get_or_create_collection("enterprise_documents", metadata={"hnsw:space": "cosine"})
app = fresh_app(path, 0)
with app.app_context():
    vc.get_collection(vc.get_chroma_client())
    assert ef_search_of(path) == 100
print("  ✅ PASS\n")

print("🎉 TẤT CẢ TEST EF_SEARCH PASS!")
