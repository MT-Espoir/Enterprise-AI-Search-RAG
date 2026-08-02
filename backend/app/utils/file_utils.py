import os
import uuid
from werkzeug.utils import secure_filename

def get_file_extension(filename: str) -> str:
    """Lấy đuôi file (vd: .pdf, .docx), viết thường"""
    return os.path.splitext(filename)[1].lower()

def get_file_size_kb(file_path: str) -> float:
    """Lấy kích thước file tính bằng KB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / 1024.0
    return 0.0

def validate_file(file, allowed_extensions: str) -> tuple[bool, str]:
    """Kiểm tra loại file"""
    if not file:
        return False, "Không tìm thấy file upload."
    
    filename = secure_filename(file.filename)
    if not filename:
        return False, "Tên file không hợp lệ."
        
    ext = get_file_extension(filename)
    allowed_list = [f".{x.strip().lower()}" for x in allowed_extensions.split(",")]
    
    if ext not in allowed_list:
        return False, f"Chỉ hỗ trợ các định dạng: {allowed_extensions}."
        
    return True, ""

def save_file(file, upload_folder: str) -> dict:
    """Lưu file vật lý và trả về thông tin"""
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        
    original_name = secure_filename(file.filename)
    extension = get_file_extension(original_name)
    
    # Sinh tên file ngẫu nhiên UUID để không bị trùng (vd: a1b2c3d4.pdf)
    uuid_filename = f"{uuid.uuid4().hex}{extension}"
    file_path = os.path.join(upload_folder, uuid_filename)
    
    # Lệnh lưu file của werkzeug FileStorage
    file.save(file_path)
    
    return {
        "uuid_filename": uuid_filename,
        "original_name": original_name,
        "extension": extension.replace(".", ""),
        "file_path": file_path,
        "size_kb": get_file_size_kb(file_path)
    }

def delete_file(file_path: str) -> bool:
    """Xóa file vật lý khỏi hệ thống"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except Exception:
        return False
