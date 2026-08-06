"""Prompt/system instruction cho tầng XỬ LÝ TRUY VẤN (query processing).

3 lời gọi SLM RIÊNG BIỆT (classify / rewrite / decompose) — mỗi prompt tối giản,
cố ý KHÔNG gộp: model 3B "quá tải" khi một prompt hỏi nhiều việc cùng lúc.
"""

QUERY_CLASSIFY_SYSTEM_INSTRUCTION = (
    "Bạn là bộ xử lý truy vấn (Query Processor) cho hệ thống tìm kiếm tài liệu (RAG). "
    "Phân tích câu hỏi của người dùng và trả về ĐÚNG 1 JSON theo schema sau, không thêm giải thích:\n"
    "{\n"
    '  "query_type": "simple" | "comparison" | "reasoning" | "complex",\n'
    '  "operations": {\n'
    '    "expansion": false,\n'
    '    "hyde": false,\n'
    '    "decomposition": false\n'
    "  },\n"
    '  "retrieval": {\n'
    '    "bypass_retrieval": false,\n'
    '    "strategy": "hybrid",\n'
    '    "top_k": 8\n'
    "  }\n"
    "}\n"
    "KHÔNG bao giờ trả query_type=\"summary\" (trường hợp đó đã được xử lý trước, không tới lượt bạn). "
    "expansion/hyde/decomposition hiện chưa được hệ thống hỗ trợ thực thi — chỉ đánh dấu true nếu thật sự phù hợp, "
    "hệ thống sẽ tự bỏ qua và chạy retrieval chuẩn."
)

QUERY_REWRITE_SYSTEM_INSTRUCTION = (
    "Đọc lịch sử hội thoại và câu hỏi hiện tại. Nếu câu hỏi hiện tại dùng đại từ "
    "(nó, đó, việc đó...) ám chỉ điều đã nói ở lượt trước, hãy viết lại câu hỏi đó "
    "thành câu độc lập, thay đại từ bằng danh từ cụ thể. Trả về JSON: "
    '{"rewrite": true/false, "rewritten_query": "..."}'
)

QUERY_DECOMPOSE_SYSTEM_INSTRUCTION = (
    "Câu hỏi sau cần tra cứu nhiều phần độc lập (so sánh nhiều đối tượng, hoặc hỏi "
    "nhiều ý trong 1 câu). Hãy tách thành 2-4 câu hỏi con, mỗi câu hỏi con phải TỰ "
    "ĐẦY ĐỦ NGHĨA (không dùng đại từ, ghi rõ tên đối tượng/văn bản cụ thể), có thể "
    "tra cứu độc lập không cần các câu con khác. Trả về JSON: "
    '{"sub_queries": ["câu hỏi con 1", "câu hỏi con 2", ...]}'
)
