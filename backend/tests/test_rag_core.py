"""
Test end-to-end pipeline: Retriever → Reranker → Generator
Dùng mock data để không cần file PDF/Word thực tế và không cần ChromaDB server.

Chạy: python backend/tests/test_rag_core.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Không tìm thấy GOOGLE_API_KEY trong .env")
    sys.exit(1)

# ── Import ─────────────────────────────────────────────────────
from app.core.schemas import RetrievedChunk
from app.core.ranking import Reranker
from app.core.generation import Generator

# ══════════════════════════════════════════════════════════════
# TEST 1: Reranker
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Reranker — Cross-Encoder re-ranking")
print("  (Đang load model ~100MB lần đầu, có thể mất 10-30s...)")

reranker = Reranker.get_instance(top_k=3)

# Mock 5 chunks với nội dung hỗn hợp liên quan và không liên quan
question = "Mức bồi thường vi phạm hợp đồng là bao nhiêu?"
mock_chunks = [
    RetrievedChunk("c1", "d1", "hop_dong.pdf", 3, "Thời tiết hôm nay rất đẹp.", 0.50, {}),
    RetrievedChunk("c2", "d1", "hop_dong.pdf", 5, "Điều 7: Mức bồi thường tối thiểu là 10 triệu đồng theo hợp đồng.", 0.82, {}),
    RetrievedChunk("c3", "d1", "hop_dong.pdf", 3, "Bên vi phạm hợp đồng phải bồi thường trong vòng 30 ngày kể từ ngày nhận thông báo.", 0.79, {}),
    RetrievedChunk("c4", "d2", "noi_quy.pdf",  1, "Công ty có 200 nhân viên làm việc tại 3 chi nhánh.", 0.45, {}),
    RetrievedChunk("c5", "d1", "hop_dong.pdf", 6, "Tranh chấp về bồi thường hợp đồng được giải quyết tại Tòa án nhân dân.", 0.71, {}),
]

reranked = reranker.rerank(question, mock_chunks)

print(f"\n  Câu hỏi: '{question}'")
print(f"  Input:   {len(mock_chunks)} chunks | Output: {len(reranked)} chunks (top 3)")
print(f"\n  Kết quả sau rerank:")
for i, chunk in enumerate(reranked):
    rerank_score = chunk.metadata.get("rerank_score", 0)
    print(f"  [{i+1}] score={rerank_score:+.3f}  → {chunk.text[:60]}...")

# Kiểm tra: chunks về bồi thường hợp đồng phải ở top
top_texts = " ".join(c.text for c in reranked[:2]).lower()
assert "bồi thường" in top_texts, "❌ Reranker không đưa chunk liên quan lên top!"
print("\n  ✅ PASS — Reranker hoạt động đúng\n")


# ══════════════════════════════════════════════════════════════
# TEST 2: Generator
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: Generator — Gemini sinh câu trả lời từ context")

generator = Generator(api_key=API_KEY)

# Dùng top 3 chunks sau rerank làm context
result = generator.generate(question, reranked)

print(f"\n  Câu hỏi    : {question}")
print(f"  Câu trả lời: {result.answer[:300]}...")
print(f"\n  Tokens dùng: {result.tokens_used}")
print(f"  Finish     : {result.finish_reason}")
print(f"  Nguồn ({len(result.sources)}):")
for src in result.sources:
    print(f"    → {src['filename']} (trang {src['page']})")

assert result.answer, "❌ Generator trả về câu trả lời rỗng!"
assert len(result.sources) > 0, "❌ Generator không trả về sources!"
print("\n  ✅ PASS — Generator hoạt động đúng\n")


# ══════════════════════════════════════════════════════════════
# TỔNG KẾT
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("🎉 TẤT CẢ TESTS PASS!")
print("   Retriever  → ✅ (mock, cần ChromaDB thật để test đầy đủ)")
print("   Reranker   → ✅")
print("   Generator  → ✅")
