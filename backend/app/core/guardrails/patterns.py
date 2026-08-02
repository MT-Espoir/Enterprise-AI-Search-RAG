"""
Regex/heuristic dùng chung cho Tier 1 Guardrails (Input/Retrieval/Output).

Đây là pattern-match bề mặt (surface pattern-matching), KHÔNG phải semantic
detector — Tier 1 dùng regex thuần theo đúng phong cách query_processor.py
(_REFERENTIAL_PATTERN, _COMPLEX_SIGNAL_PATTERN), không thêm dependency ML nặng.
"""
import re

# ── Prompt Injection (VN + EN) — HARD BLOCK
# Nhắm vào cụm cố gắng ghi đè/vô hiệu hoá system prompt. Giữ cụm cụ thể (không
# chỉ 1 từ đơn lẻ như "ignore" hay "bỏ qua" — quá phổ biến trong câu hỏi luật
# hợp pháp, vd "bỏ qua điều khoản này có được không?").
PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)\b",
    r"\bdisregard\s+(all\s+|any\s+)?(previous|prior|above)\s+(instructions?|rules?)\b",
    r"\boverride\s+(the\s+)?(system\s+)?(prompt|instructions?|rules?)\b",
    r"\breveal\s+(your|the)\s+(system\s+prompt|instructions?)\b",
    r"\bnew\s+system\s+prompt\b",
    r"\byou\s+are\s+now\s+(a|an)\b",
    r"\bact\s+as\s+(if\s+you\s+(are|were)\s+)?(an?\s+)?(unrestricted|uncensored|jailbroken)\b",
    r"\bbỏ\s+qua\s+(mọi|tất cả|toàn bộ)\s+(hướng dẫn|chỉ dẫn|quy tắc|lệnh|prompt)\b",
    r"\bquên\s+(đi\s+)?(mọi|tất cả)?\s*(hướng dẫn|vai trò|quy tắc)\s+(trước đó|đã (được )?cho|ban đầu)\b",
    r"\bbỏ\s+vai\s+trò\s+(trợ lý|ai)\b",
    r"\bkhông\s+(cần|phải)\s+tuân\s+theo\s+(quy tắc|hướng dẫn|system prompt)\b",
    r"\btiết\s+lộ\s+(system\s+)?prompt\b",
    r"\bhệ\s+thống\s+prompt\s+mới\b",
    r"\bbạn\s+(bây giờ|giờ)\s+là\s+(một\s+)?(ai|trợ lý)\s+(mới|khác|không giới hạn)\b",
    r"\bin\s+developer\s+mode\b",
    r"\bchế\s+độ\s+(nhà\s+)?phát\s+triển\b",
]
_PROMPT_INJECTION_PATTERN = re.compile("|".join(PROMPT_INJECTION_PATTERNS), re.IGNORECASE)

# ── Jailbreak (roleplay-based rule bypass) — HARD BLOCK 
JAILBREAK_PATTERNS = [
    r"\bdo\s+anything\s+now\b",
    # "DAN" đứng riêng dễ trùng tên người -> chỉ khớp khi kèm ngữ cảnh jailbreak
    # trong phạm vi 40 ký tự lân cận.
    r"\bDAN\b(?=.{0,40}(jailbreak|do anything|no rules|unrestricted))",
    r"\bjailbreak(ed|ing)?\b",
    r"\bhypothetical(ly)?\b.{0,30}\b(no rules?|without (any )?restrictions?|unrestricted)\b",
    r"\bpretend\s+(that\s+)?you\s+(have\s+no|don't have any)\s+(rules|restrictions|filters)\b",
    r"\bkhông\s+bị\s+(giới hạn|ràng buộc)\s+(bởi\s+)?(quy tắc|đạo đức|chính sách|kiểm duyệt)\b",
    r"\bvượt\s+qua\s+(mọi\s+)?(giới hạn|kiểm duyệt|bộ lọc)\b",
    r"\bgiả\s+(vờ|sử)\s+(bạn\s+)?(không|ko)\s+có\s+(bất kỳ\s+)?(quy tắc|giới hạn|ràng buộc)\b",
    r"\btrả\s+lời\s+như\s+thể\s+(bạn\s+)?(không|ko)\s+có\s+(quy tắc|ràng buộc)\b",
]
_JAILBREAK_PATTERN = re.compile("|".join(JAILBREAK_PATTERNS), re.IGNORECASE)

# ── PII (Vietnam-specific) — FLAG/LOG ONLY, KHÔNG BAO GIỜ chặn 
_CCCD_CMND_PATTERN = re.compile(r"\b\d{9}(?:\d{3})?\b")

# SĐT di động VN: 0 hoặc +84 + đầu số nhà mạng hợp lệ + 7 số còn lại.
_PHONE_PATTERN = re.compile(r"(?:\+84|0)(?:3[2-9]|5[689]|7[06-9]|8[1-9]|9[0-9])\d{7}\b")

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# ── Citation cross-check (Output Guard) ─────────────────────────────────────
# Khớp "Điều <số>[chữ cái phụ]" — đúng định dạng heading luật VN mà
# RecursiveChunker đã tách theo (xem architecture_diagram.md), nên nếu 1 chunk
# thật sự chứa Điều đó, cụm "Điều <số>" sẽ xuất hiện y hệt trong chunk.text.

_CITATION_PATTERN = re.compile(r"Điều\s+\d+[a-zđ]?\b", re.IGNORECASE)
