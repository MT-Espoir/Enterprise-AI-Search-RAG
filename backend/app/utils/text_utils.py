import re
import unicodedata

def clean_text(text: str) -> str:
    """Làm sạch văn bản: xóa ký tự đặc biệt, chuẩn hóa khoảng trắng"""
    if not text:
        return ""
    
    # Chuẩn hóa Unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Xóa các ký tự control ẩn (\x00, \x0c...)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"))
    
    # Nhiều khoảng trắng/tab liền nhau -> 1 khoảng trắng
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Nhiều dòng trống liền nhau -> Tối đa 2 dòng trống
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def truncate_text(text: str, max_chars: int = 100, suffix: str = "...") -> str:
    """Cắt ngắn văn bản để làm tiêu đề hoặc hiển thị snippet"""
    if len(text) <= max_chars:
        return text
    
    # Tìm khoảng trắng gần nhất để không cắt ngang giữa một chữ
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
        
    return truncated + suffix

def generate_session_title(question: str, max_len: int = 50) -> str:
    """Tạo title cho chat session từ câu hỏi đầu tiên"""
    # Xóa các dấu câu và kí tự đặc biệt
    clean_q = re.sub(r'[^\w\s]', '', question)
    return truncate_text(clean_q, max_len)

def count_words(text: str) -> int:
    """Đếm số từ xấp xỉ trong text"""
    return len(text.split())
