"""
Dashboard API — Layer 5 Observability (Phase 5, xem
rag_core_quality_roadmap.md mục 6m). Admin-only, đọc dữ liệu đã persist qua
`request_tracer.py` (collection `request_logs`) và `Document.ocr_stats`
(PDFParser, xem pipeline.py) — KHÔNG tính toán lại gì mới, chỉ aggregate.
"""
from datetime import datetime, timedelta, timezone

from flask import Blueprint, request

from ..extensions import mongo
from ..models.document import Document
from ..core.observability.request_tracer import REQUEST_LOGS_COLLECTION
from ..utils.auth_decorators import admin_required
from ..utils.response_utils import success_response, error_response

dashboard_bp = Blueprint("dashboard", __name__)

_LATENCY_FIELDS = (
    "preprocessing_ms", "retrieval_ms", "rerank_ms", "generation_ms", "total_ms",
)


@dashboard_bp.route("/latency", methods=["GET"])
@admin_required
def latency_over_time():
    """Latency trung bình theo giờ, N ngày gần nhất (mặc định 7, tối đa 30)."""
    try:
        days = min(int(request.args.get("days", 7)), 30)
    except ValueError:
        return error_response("Tham số 'days' phải là số nguyên.")

    since = datetime.now(timezone.utc) - timedelta(days=days)
    group_stage = {"_id": {"$dateToString": {"format": "%Y-%m-%dT%H:00:00Z", "date": "$timestamp"}}, "count": {"$sum": 1}}
    for field in _LATENCY_FIELDS:
        group_stage[f"avg_{field}"] = {"$avg": f"$latency.{field}"}

    pipeline = [
        {"$match": {"timestamp": {"$gte": since}}},
        {"$group": group_stage},
        {"$sort": {"_id": 1}},
    ]
    rows = list(mongo.db[REQUEST_LOGS_COLLECTION].aggregate(pipeline))
    buckets = []
    for r in rows:
        bucket = {"bucket": r["_id"], "count": r["count"]}
        for field in _LATENCY_FIELDS:
            val = r.get(f"avg_{field}")
            bucket[f"avg_{field}"] = round(val, 1) if val is not None else None
        buckets.append(bucket)

    return success_response(data={"buckets": buckets, "days": days})


@dashboard_bp.route("/ocr", methods=["GET"])
@admin_required
def ocr_fail_rate():
    """Tỷ lệ OCR fail — tổng hợp + theo từng document (Document.ocr_stats,
    chỉ document PDF từng qua nhánh cần-OCR mới có field này)."""
    docs = list(mongo.db[Document.COLLECTION].find(
        {"ocr_stats": {"$ne": None}},
        {"original_name": 1, "ocr_stats": 1},
    ))

    total_needing = sum(d["ocr_stats"].get("pages_needing_ocr", 0) for d in docs)
    total_attempted = sum(d["ocr_stats"].get("pages_ocr_attempted", 0) for d in docs)
    total_failed = sum(d["ocr_stats"].get("pages_ocr_failed", 0) for d in docs)

    per_document = []
    for d in docs:
        s = d["ocr_stats"]
        needing = s.get("pages_needing_ocr", 0)
        if needing == 0:
            continue
        failed = s.get("pages_ocr_failed", 0)
        per_document.append({
            "doc_id": str(d["_id"]),
            "filename": d.get("original_name"),
            "pages_needing_ocr": needing,
            "pages_ocr_failed": failed,
            "fail_rate": round(failed / needing, 3),
        })
    per_document.sort(key=lambda x: x["fail_rate"], reverse=True)

    return success_response(data={
        "overall_fail_rate": round(total_failed / total_needing, 3) if total_needing else None,
        "total_pages_needing_ocr": total_needing,
        "total_pages_ocr_attempted": total_attempted,
        "total_pages_ocr_failed": total_failed,
        "documents_with_ocr": len(per_document),
        "per_document": per_document,
    })


@dashboard_bp.route("/summary", methods=["GET"])
@admin_required
def summary():
    """Tổng quan nhanh cho đầu trang Dashboard."""
    since_24h = datetime.now(timezone.utc) - timedelta(hours=24)
    total_requests = mongo.db[REQUEST_LOGS_COLLECTION].count_documents({})
    requests_24h = mongo.db[REQUEST_LOGS_COLLECTION].count_documents({"timestamp": {"$gte": since_24h}})

    agg = list(mongo.db[REQUEST_LOGS_COLLECTION].aggregate([
        {"$match": {"timestamp": {"$gte": since_24h}}},
        {"$group": {"_id": None, "avg_total_ms": {"$avg": "$latency.total_ms"}}},
    ]))
    avg_latency_24h = round(agg[0]["avg_total_ms"], 1) if agg and agg[0]["avg_total_ms"] is not None else None

    return success_response(data={
        "total_requests": total_requests,
        "requests_last_24h": requests_24h,
        "avg_latency_ms_last_24h": avg_latency_24h,
    })
