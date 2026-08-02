import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env để lấy GOOGLE_API_KEY
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Không tìm thấy GOOGLE_API_KEY trong file .env")
    sys.exit(1)

# ── Import ────────────────────────────────────────────────────
from app.ingestion.embedder.google_embedder import GoogleEmbedder

# ── Khởi tạo Embedder ─────────────────────────────────────────
print("🔧 Khởi tạo GoogleEmbedder...")
embedder = GoogleEmbedder(api_key=API_KEY)
print(f"   Model: {embedder.model_name}\n")

# ── Test 1: Embed 1 đoạn văn tài liệu ────────────────────────
print("=" * 60)
print("TEST 1: embed_document()")
doc_text = "Hợp đồng lao động bị vi phạm điều khoản bồi thường thiệt hại."
vec = embedder.embed_document(doc_text)
print(f"  Input : {doc_text}")
print(f"  Vector length : {len(vec)}")
print(f"  First 5 dims  : {[round(v, 4) for v in vec[:5]]}")

# ── Test 2: Embed 1 câu hỏi ────────────────────────────────────
print("\nTEST 2: embed_query()")
query = "Bồi thường vi phạm hợp đồng là bao nhiêu?"
vec_q = embedder.embed_query(query)
print(f"  Input : {query}")
print(f"  Vector length : {len(vec_q)}")
print(f"  First 5 dims  : {[round(v, 4) for v in vec_q[:5]]}")

# ── Test 3: Cosine Similarity (tay) ────────────────────────────
print("\nTEST 3: Cosine Similarity")
import math

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a, b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    return dot / (mag_a * mag_b)

similar_text = "Điều khoản bồi thường trong hợp đồng lao động."
unrelated_text = "Thời tiết hôm nay rất đẹp, trời nắng và mát."

vec_sim = embedder.embed_document(similar_text)
vec_unr = embedder.embed_document(unrelated_text)

score_same    = cosine_sim(vec, vec_q)
score_similar = cosine_sim(vec, vec_sim)
score_diff    = cosine_sim(vec, vec_unr)

print(f"  doc ↔ query (liên quan)       : {score_same:.4f}  ✅")
print(f"  doc ↔ similar_doc (gần nghĩa) : {score_similar:.4f}  ✅")
print(f"  doc ↔ unrelated (khác chủ đề) : {score_diff:.4f}  ❌")
print(f"\n  Kỳ vọng: score_same ≈ score_similar >> score_diff")

# ── Test 4: embed_batch ────────────────────────────────────────
print("\nTEST 4: embed_batch() với 3 chunks")
batch = [
    "Điều 5: Bên A phải bồi thường trong vòng 30 ngày.",
    "Điều 6: Mức bồi thường tối thiểu là 10 triệu đồng.",
    "Điều 7: Tranh chấp được giải quyết tại Tòa án nhân dân."
]
vecs = embedder.embed_batch(batch)
print(f"  Số chunk đầu vào  : {len(batch)}")
print(f"  Số vector đầu ra  : {len(vecs)}")
print(f"  Mỗi vector có     : {len(vecs[0])} dimensions")
print(f"\n✅ Tất cả tests hoàn thành!")