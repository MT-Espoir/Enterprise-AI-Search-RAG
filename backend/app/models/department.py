# Danh sách phòng ban cố định
DEPARTMENTS = (
    "ban_giam_doc",   # Ban Giám đốc
    "hr",             # Phòng Nhân sự (HR)
    "finance",        # Phòng Tài chính - Kế toán
    "sales",          # Phòng Kinh doanh (Sales)
    "marketing",      # Phòng Marketing
    "it",             # Phòng Công nghệ thông tin (IT)
    "rd",             # Phòng Nghiên cứu và Phát triển (R&D)
    "operations",     # Phòng Vận hành / Sản xuất
    "procurement",    # Phòng Mua hàng & Chuỗi cung ứng
    "legal",          # Phòng Pháp chế & Tuân thủ (Legal)
    "cs",             # Phòng Chăm sóc khách hàng (CS)
    "qa_qc",          # Phòng Đảm bảo chất lượng (QA/QC)
)

# "" = chưa gán phòng ban = MỞ cho tất cả
# KHÔNG dùng None vì ChromaDB metadata không nhận giá trị None.
DEFAULT_DEPARTMENT = ""

DEPARTMENT_LABELS = {
    "ban_giam_doc": "Ban Giám đốc",
    "hr": "Phòng Nhân sự (HR)",
    "finance": "Phòng Tài chính - Kế toán",
    "sales": "Phòng Kinh doanh (Sales)",
    "marketing": "Phòng Marketing",
    "it": "Phòng Công nghệ thông tin (IT)",
    "rd": "Phòng Nghiên cứu và Phát triển (R&D)",
    "operations": "Phòng Vận hành / Sản xuất",
    "procurement": "Phòng Mua hàng & Chuỗi cung ứng",
    "legal": "Phòng Pháp chế & Tuân thủ (Legal)",
    "cs": "Phòng Chăm sóc khách hàng (CS)",
    "qa_qc": "Phòng Đảm bảo chất lượng (QA/QC)",
}
