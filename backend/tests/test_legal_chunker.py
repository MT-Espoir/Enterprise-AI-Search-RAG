"""
Test RecursiveChunker — separator Chương/Điều-aware cho văn bản luật VN.

Bối cảnh: chunker gốc chỉ tách theo \\n\\n/\\n/space/char, có thể cắt ngang 1 Điều
(quy định chính ở Khoản đầu, ngoại lệ ở Khoản cuối bị tách sang chunk khác).
Thêm 2 separator regex (Chương/Điều) ưu tiên tách TRƯỚC — chỉ khớp từ khóa
tiếng Việt nên KHÔNG ảnh hưởng tài liệu không phải văn bản luật (xem test 3,
so sánh byte-for-byte với tài liệu benchmark tiếng Anh hiện có).

Chạy: python backend/tests/test_legal_chunker.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.chunker.recursive_chunker import RecursiveChunker
from app.ingestion.parsers.markdown_parser import MarkdownParser

# ══════════════════════════════════════════════════════════════
# TEST 1: Điều vừa chunk_size → giữ nguyên trong 1 chunk (không cắt ngang)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Điều đủ nhỏ để nằm gọn 1 chunk, kể cả Khoản ngoại lệ")

text = (
    "Điều 5. Điều kiện cấp giấy phép\n"
    "1. Tổ chức, cá nhân phải có trụ sở hợp pháp.\n"
    "2. Hồ sơ gồm đơn đề nghị và giấy tờ liên quan.\n"
    "3. Ngoại lệ: không áp dụng khoản 1 nếu đã có giấy phép tương đương theo điều ước quốc tế.\n\n"
    "Điều 6. Thời hạn hiệu lực\n"
    "Giấy phép có thời hạn 5 năm."
)

chunker = RecursiveChunker(chunk_size=250, chunk_overlap=0)
chunks = chunker.chunk_text(text)
print(f"  Số chunk: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"  chunk {i}: {c[:60]!r}...")

assert any("Ngoại lệ" in c and "Điều 5" in c for c in chunks), (
    "❌ Khoản ngoại lệ (khoản 3) bị tách khỏi Điều 5 dù đủ chỗ trong chunk_size"
)
assert any(c.startswith("Điều 6.") for c in chunks), "❌ Điều 6 phải bắt đầu 1 chunk mới, không dính vào Điều 5"
print("  ✅ PASS\n")


# ══════════════════════════════════════════════════════════════
# TEST 2: Chương boundary cũng được ưu tiên tách
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: Tách ưu tiên tại ranh giới Chương")

text2 = (
    "Chương I. QUY ĐỊNH CHUNG\n"
    "Điều 1. Phạm vi điều chỉnh\nNội dung điều 1.\n\n"
    "Chương II. ĐIỀU KHOẢN THI HÀNH\n"
    "Điều 10. Hiệu lực thi hành\nNội dung điều 10."
)
chunker2 = RecursiveChunker(chunk_size=60, chunk_overlap=0, min_chunk_length=1)
chunks2 = chunker2.chunk_text(text2)
print(f"  chunks: {chunks2}")
assert any(c.startswith("Chương II") for c in chunks2), "❌ Phải có 1 chunk bắt đầu đúng tại 'Chương II'"
print("  ✅ PASS\n")


# ══════════════════════════════════════════════════════════════
# TEST 3: No-op tuyệt đối với tài liệu KHÔNG phải văn bản luật (tiếng Anh)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: Không đổi hành vi với tài liệu benchmark tiếng Anh hiện có")


class OldChunker(RecursiveChunker):
    """Tái tạo đúng cấu hình splitter TRƯỚC khi thêm separator Chương/Điều."""
    def __init__(self, chunk_size=1000, chunk_overlap=200, min_chunk_length=50):
        self.min_chunk_length = min_chunk_length
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )


TEST_FILE = os.path.join(os.path.dirname(__file__), "Test_subject/System_Design_Deep_Feature_Analysis.md")
pages = MarkdownParser().parse(TEST_FILE)

old_chunks = OldChunker(chunk_size=1000, chunk_overlap=200).chunk_pages(pages)
new_chunks = RecursiveChunker(chunk_size=1000, chunk_overlap=200).chunk_pages(pages)

print(f"  old count={len(old_chunks)} | new count={len(new_chunks)}")
assert old_chunks == new_chunks, "❌ Thêm separator Chương/Điều làm đổi chunking của tài liệu KHÔNG phải văn bản luật!"
print("  ✅ PASS — byte-for-byte giống hệt, không hồi quy tài liệu hiện có\n")

print("=" * 60)
print("🎉 TEST LEGAL CHUNKER HOÀN THÀNH — TẤT CẢ PASS!")
