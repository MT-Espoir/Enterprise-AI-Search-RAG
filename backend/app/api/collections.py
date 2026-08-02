# ==============================================================
# app/api/collections.py — Collection Management Endpoints
# ==============================================================
# Blueprint prefix: /api/collections
# Tất cả endpoints yêu cầu JWT authentication (Admin only)
#
# Lưu ý: Hệ thống dùng SHARED COLLECTION (tất cả user dùng chung)
# File này dùng để quản lý metadata collection, không phân chia theo user.
#
# Endpoints:
#   GET  /              → Thông tin collection hiện tại (số documents, vectors)
#                         Return: 200 + { name, document_count, vector_count }
#
#   POST /reset         → Xóa toàn bộ vectors trong collection (Admin only)
#                         Return: 200 + { message }
#
#   GET  /stats         → Thống kê chi tiết về collection
#                         Return: 200 + { total_chunks, avg_chunks_per_doc, ... }
# ==============================================================

from flask import Blueprint
from ..utils.auth_decorators import admin_required
from ..utils.response_utils import success_response
from ..extensions import mongo
from ..models.document import Document
from ..vectorstore.operations import VectorStoreOps

collections_bp = Blueprint("collections", __name__)

@collections_bp.route("/", methods=["GET"])
@admin_required
def get_collection_info():
    vector_ops = VectorStoreOps()
    document_count = mongo.db[Document.COLLECTION].count_documents({})
    vector_count = vector_ops.get_collection_count()
    return success_response(data={
        "name": vector_ops.collection.name,
        "document_count": document_count,
        "vector_count": vector_count,
    })

# @collections_bp.route("/reset", methods=["POST"])
# @jwt_required()
# def reset_collection(): ...

# @collections_bp.route("/stats", methods=["GET"])
# @jwt_required()
# def get_stats(): ...
