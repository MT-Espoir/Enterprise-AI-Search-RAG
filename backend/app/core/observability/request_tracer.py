import logging
from datetime import datetime, timezone

import structlog
from flask import has_app_context

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(ensure_ascii=False),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

_tracer = structlog.get_logger("rag_pipeline")
_fallback_logger = logging.getLogger(__name__)

REQUEST_LOGS_COLLECTION = "request_logs"


def log_rag_request(
    request_id: str,
    question: str,
    query_plan: dict,
    latency: dict,
    retrieval: dict,
    guardrail: dict = None,
    enabled: bool = True,
) -> None:
    if not enabled:
        return

    _tracer.info(
        "rag_request",
        request_id=request_id,
        query=question,
        query_plan=query_plan,
        latency=latency,
        retrieval=retrieval,
        guardrail=guardrail,
    )

    _persist_to_mongo(request_id, question, query_plan, latency, retrieval, guardrail)


def _persist_to_mongo(request_id, question, query_plan, latency, retrieval, guardrail=None) -> None:
    """Fail-open TUYỆT ĐỐI: benchmark script gọi RAGPipeline trực tiếp không
    qua Flask (không có app context) hoặc Mongo lỗi đều KHÔNG được làm crash
    request RAG thật — chỉ log cảnh báo qua `logging` chuẩn."""
    if not has_app_context():
        return
    try:
        from ...extensions import mongo
        mongo.db[REQUEST_LOGS_COLLECTION].insert_one({
            "request_id": request_id,
            "query": question,
            "query_plan": query_plan,
            "latency": latency,
            "retrieval": retrieval,
            "guardrail": guardrail,
            "timestamp": datetime.now(timezone.utc),
        })
    except Exception as exc:
        _fallback_logger.warning(f"Không lưu được request_log vào Mongo: {exc}")
