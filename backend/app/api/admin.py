from bson.errors import InvalidId
from flask import Blueprint, current_app, request
from flask_jwt_extended import get_jwt_identity

from ..core.retrieval import get_bm25_index
from ..ingestion.embedder.local_embedder import LocalEmbedder
from ..services.context_header_reembed import (
    DEFAULT_BATCH_SIZE, ContextHeaderReembedder, JobAlreadyRunning, ReembedJobStore,
)
from ..utils.auth_decorators import admin_required
from ..utils.response_utils import error_response, success_response
from ..vectorstore.operations import VectorStoreOps

admin_bp = Blueprint("admin", __name__)


def _embedder():
    # Tách hàm để test thay embedder giả (ChromaDB test dùng vector 3 chiều)
    return LocalEmbedder.get_instance(model_name=current_app.config.get("LOCAL_EMBEDDING_MODEL"))


def _reembedder(batch_size: int | None = None, with_embedder: bool = True) -> ContextHeaderReembedder:
    return ContextHeaderReembedder(
        VectorStoreOps(),
        _embedder() if with_embedder else None,
        bm25_index=get_bm25_index(),
        batch_size=batch_size or current_app.config.get("CONTEXT_HEADER_REEMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE),
    )


@admin_bp.route("/context-header/reembed", methods=["GET"])
@admin_required
def reembed_status():
    """Tiến độ job embed lại header ngữ cảnh (đọc MongoDB — worker nào trả lời cũng như nhau).
    ?preview=1 kèm danh sách tài liệu đang chờ embed lại."""
    data = {"job": ReembedJobStore().get()}
    if request.args.get("preview") in ("1", "true"):
        data["pending"] = _reembedder(with_embedder=False).preview()
    return success_response(data)


@admin_bp.route("/context-header/reembed", methods=["POST"])
@admin_required
def start_reembed():
    """Body (tùy chọn): {"doc_ids": [...], "batch_size": 16, "dry_run": true}.
    dry_run → chỉ trả kế hoạch (số tài liệu/chunk sẽ embed), không ghi gì.
    Ngược lại chạy nền trong tiến trình server → 202; đang có job chạy → 409.
    Job CHIẾM CPU (BGE-M3 ~0,9 chunk/s trên CPU) — chạy ngoài giờ."""
    data = request.get_json(silent=True) or {}
    doc_ids = data.get("doc_ids")
    if doc_ids is not None and (not isinstance(doc_ids, list) or not all(isinstance(i, str) for i in doc_ids)):
        return error_response("'doc_ids' phải là danh sách chuỗi.")
    batch_size = data.get("batch_size")
    if batch_size is not None and (not isinstance(batch_size, int) or not 1 <= batch_size <= 128):
        return error_response("'batch_size' phải là số nguyên 1–128.")
    try:
        if data.get("dry_run") is True:
            return success_response(_reembedder(with_embedder=False).preview(doc_ids))
        _reembedder(batch_size).start_async(doc_ids, started_by=get_jwt_identity())
    except InvalidId:
        return error_response("'doc_ids' chứa ID không hợp lệ.")
    except JobAlreadyRunning as exc:
        return error_response(str(exc), error_code="job_running", status_code=409)
    return success_response({"job": ReembedJobStore().get()}, message="Đã bắt đầu embed lại.", status_code=202)


@admin_bp.route("/context-header/reembed/cancel", methods=["POST"])
@admin_required
def cancel_reembed():
    """Dừng sau lô đang chạy; chunk đã ghi giữ nguyên, lần chạy sau làm tiếp phần còn lại."""
    if not ReembedJobStore().request_cancel():
        return error_response("Không có job nào đang chạy.", status_code=409)
    return success_response({"job": ReembedJobStore().get()}, message="Đã yêu cầu dừng.", status_code=202)
