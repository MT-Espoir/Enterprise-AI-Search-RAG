import json
import logging
import re
from dataclasses import dataclass, field

import requests

from .prompts import (
    QUERY_CLASSIFY_SYSTEM_INSTRUCTION,
    QUERY_DECOMPOSE_SYSTEM_INSTRUCTION,
    QUERY_REWRITE_SYSTEM_INSTRUCTION,
)

logger = logging.getLogger(__name__)

# ── Fast Path (Rule Engine) — hợp nhất từ IntentRouter + QueryAnalyzer cũ ──

SUMMARY_KEYWORDS = [
    r"tóm tắt",
    r"tổng quan",
    r"tổng hợp",
    r"nội dung chính",
    r"nói về cái gì",
    r"nói về vấn đề gì",
    r"tóm ý",
    r"chủ đề chính",
    r"summarize",
    r"summary",
    r"overview",
]
_SUMMARY_PATTERN = re.compile("|".join(SUMMARY_KEYWORDS), re.IGNORECASE)

_REFERENTIAL_PATTERN = re.compile(
    r"\b(nó|họ|chúng nó|cái đó|cái này|như trên|như vậy|ở trên|vừa nói|vừa rồi|trên đó|"
    r"đó|này|việc đó|văn bản đó|điều kiện đó|khoản đó)\b",
    re.IGNORECASE,
)

SHORT_QUESTION_WORD_THRESHOLD = 8

_COMPLEX_SIGNAL_PATTERN = re.compile(
    r"\b(so sánh|khác nhau|giống nhau|so với|khác gì)\b",
    re.IGNORECASE,
)
LONG_QUESTION_WORD_THRESHOLD = 20


_UNIMPLEMENTED_OPERATIONS = ("expansion", "hyde", "decomposition")

CLASSIFY_SYSTEM_INSTRUCTION = QUERY_CLASSIFY_SYSTEM_INSTRUCTION
REWRITE_SYSTEM_INSTRUCTION = QUERY_REWRITE_SYSTEM_INSTRUCTION
DECOMPOSE_SYSTEM_INSTRUCTION = QUERY_DECOMPOSE_SYSTEM_INSTRUCTION

MIN_SUB_QUERIES = 2
MAX_SUB_QUERIES = 4


@dataclass
class QueryPlan:
    """Kế hoạch xử lý 1 câu hỏi — output thống nhất của QueryProcessor."""

    query_type: str
    rewrite: bool
    rewritten_query: str | None
    bypass_retrieval: bool
    strategy_hint: str | None = None
    top_k_hint: int | None = None
    unimplemented_operations: list[str] = field(default_factory=list)
    source: str = "fast_path"  # "fast_path" | "slm" | "fallback"


def _default_plan(query_type: str = "simple", bypass_retrieval: bool = False, source: str = "fast_path") -> QueryPlan:
    return QueryPlan(
        query_type=query_type,
        rewrite=False,
        rewritten_query=None,
        bypass_retrieval=bypass_retrieval,
        source=source,
    )


class QueryProcessor:
    """
    Hợp nhất IntentRouter + QueryAnalyzer + QueryRewriter cũ thành 1 component
    duy nhất chịu trách nhiệm phân tích câu hỏi và lập QueryPlan cho Retrieval.

    Fast Path (regex, không gọi LLM) xử lý phần lớn truy vấn — chỉ gọi SLM khi
    có tín hiệu thật sự mơ hồ/phụ thuộc ngữ cảnh.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:3b-instruct",
        default_strategy: str = "hybrid",
        default_top_k: int = 10,
        slm_enabled: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.default_strategy = default_strategy
        self.default_top_k = default_top_k
        self.slm_enabled = slm_enabled

    def process(self, question: str, history: list[dict] = None) -> QueryPlan:
        fast_plan = self._try_fast_path(question, history)
        if fast_plan is not None:
            return fast_plan

        if not self.slm_enabled:
            # Killswitch: câu mơ hồ nhưng SLM bị tắt -> coi như simple, KHÔNG rewrite.
            return _default_plan(query_type="simple", bypass_retrieval=False, source="fast_path")

        plan = self._call_slm_classify(question)
        # Chỉ gọi rewrite khi có history — không có gì để giải quyết đại từ nếu
        # đây là lượt đầu tiên của hội thoại.
        if history:
            rewrite, rewritten_query = self._call_slm_rewrite(question, history)
            plan.rewrite = rewrite
            plan.rewritten_query = rewritten_query
        return plan

    # ── Fast Path (Rule Engine) ──────────────────────────────────────────

    def _try_fast_path(self, question: str, history: list[dict] = None) -> QueryPlan | None:
        if _SUMMARY_PATTERN.search(question):
            logger.info(f"QueryProcessor fast-path: SUMMARY cho câu hỏi '{question[:80]}'")
            plan = _default_plan(query_type="summary", bypass_retrieval=True, source="fast_path")
            plan.strategy_hint = self.default_strategy
            plan.top_k_hint = self.default_top_k
            return plan

        word_count = len(question.split())
        needs_slm = (
            bool(_REFERENTIAL_PATTERN.search(question))
            or (bool(history) and word_count < SHORT_QUESTION_WORD_THRESHOLD)
            or bool(_COMPLEX_SIGNAL_PATTERN.search(question))
            or word_count >= LONG_QUESTION_WORD_THRESHOLD
        )
        if needs_slm:
            return None

        plan = _default_plan(query_type="simple", bypass_retrieval=False, source="fast_path")
        plan.strategy_hint = self.default_strategy
        plan.top_k_hint = self.default_top_k
        return plan

    # ── SLM Path (2 lời gọi riêng biệt — xem docstring class) ───────────

    def _call_slm_classify(self, question: str) -> QueryPlan:
        """Lời gọi 1: query_type + expansion/hyde/decomposition + retrieval hint.
        KHÔNG nhận history — phân loại không cần ngữ cảnh hội thoại trước đó."""
        messages = [
            {"role": "system", "content": CLASSIFY_SYSTEM_INSTRUCTION},
            {"role": "user", "content": question},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            data = json.loads(content)

            operations = data.get("operations", {}) or {}
            retrieval = data.get("retrieval", {}) or {}

            unimplemented = [op for op in _UNIMPLEMENTED_OPERATIONS if operations.get(op)]
            if unimplemented:
                logger.warning(
                    f"QueryProcessor: SLM đề xuất operations chưa hỗ trợ {unimplemented} cho câu "
                    f"'{question[:80]}' — bỏ qua, chạy Standard Hybrid Retrieval."
                )

            return QueryPlan(
                query_type=data.get("query_type") or "simple",
                rewrite=False,
                rewritten_query=None,
                bypass_retrieval=bool(retrieval.get("bypass_retrieval", False)),
                strategy_hint=retrieval.get("strategy"),
                top_k_hint=retrieval.get("top_k"),
                unimplemented_operations=unimplemented,
                source="slm",
            )
        except Exception as exc:
            logger.warning(
                f"QueryProcessor SLM classify thất bại ({exc}) cho câu '{question[:80]}' — fallback về "
                f"QueryPlan mặc định (Standard Hybrid Retrieval)."
            )
            plan = _default_plan(query_type="simple", bypass_retrieval=False, source="fallback")
            plan.strategy_hint = self.default_strategy
            plan.top_k_hint = self.default_top_k
            return plan

    def _call_slm_rewrite(self, question: str, history: list[dict]) -> tuple[bool, str | None]:
        """Lời gọi 2: CHỈ quyết định rewrite — prompt tối giản, đã verify đáng tin
        cậy hơn hẳn so với gộp chung (xem rag_core_quality_roadmap.md mục 6h)."""
        messages = [{"role": "system", "content": REWRITE_SYSTEM_INSTRUCTION}]
        for msg in history:
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": question})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            data = json.loads(content)

            rewrite = bool(data.get("rewrite"))
            rewritten_query = (data.get("rewritten_query") or "").strip() or None
            if rewrite and not rewritten_query:
                # SLM báo cần rewrite nhưng không cho câu viết lại -> fail-open cục bộ.
                logger.warning(f"QueryProcessor: SLM báo rewrite=true nhưng rewritten_query rỗng cho '{question[:80]}'")
                rewrite = False
            return rewrite, (rewritten_query if rewrite else None)
        except Exception as exc:
            logger.warning(
                f"QueryProcessor SLM rewrite thất bại ({exc}) cho câu '{question[:80]}' — không rewrite."
            )
            return False, None

    def decompose(self, question: str) -> list[str]:
        """Tách câu hỏi phức tạp (query_type="complex"/"comparison"/"reasoning")
        thành các sub-query độc lập để retrieve riêng từng phần.
        Trả về list rỗng nếu không cần tách (fail-open) hoặc lỗi — caller (RAGPipeline)
        tự xử lý fallback về flow retrieval đơn giản."""
        messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_INSTRUCTION},
            {"role": "user", "content": question},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            data = json.loads(content)

            sub_queries = [str(q).strip() for q in (data.get("sub_queries") or []) if str(q).strip()]
            if not (MIN_SUB_QUERIES <= len(sub_queries) <= MAX_SUB_QUERIES):
                logger.warning(
                    f"QueryProcessor decompose: SLM trả {len(sub_queries)} sub_queries "
                    f"(ngoài khoảng [{MIN_SUB_QUERIES}, {MAX_SUB_QUERIES}]) cho '{question[:80]}' "
                    f"— fail-open, coi như không tách được."
                )
                return []
            return sub_queries
        except Exception as exc:
            logger.warning(f"QueryProcessor decompose thất bại ({exc}) cho câu '{question[:80]}' — không tách.")
            return []
