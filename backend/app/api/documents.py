import os
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from ..services.document_service import DocumentService
from ..ingestion.embedder.local_embedder import LocalEmbedder
from ..ingestion.pipeline import IngestionPipeline
from ..ingestion.ocr.tesseract_ocr_engine import TesseractOCREngine
from ..vectorstore.operations import VectorStoreOps
from ..core.bm25_index import get_bm25_index
from ..extensions import limiter, user_or_ip_key
from ..services.document_service import DuplicateDocumentError

documents_bp = Blueprint('documents', __name__, url_prefix='/api/documents')

@documents_bp.route('/upload', methods=['POST'])
@limiter.limit(lambda: current_app.config.get("RATELIMIT_UPLOAD", "10 per minute"), key_func=user_or_ip_key)
@jwt_required()
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    vector_ops = VectorStoreOps()
    doc_service = DocumentService(vector_ops, upload_folder=current_app.config['UPLOAD_FOLDER'])
    
    # LUÔN DÙNG LOCAL EMBEDDER (BGE-M3, offline, không phụ thuộc quota API)
    embedder = LocalEmbedder(model_name=current_app.config.get('LOCAL_EMBEDDING_MODEL'))

    ocr_engine = None
    if current_app.config.get('OCR_ENABLED', True):
        ocr_engine = TesseractOCREngine(
            tesseract_cmd=current_app.config.get('TESSERACT_CMD'),
            languages=current_app.config.get('OCR_LANGUAGES', 'vie+eng'),
            dpi=current_app.config.get('OCR_DPI', 200),
        )

    pipeline = IngestionPipeline(
        embedder=embedder,
        vector_ops=vector_ops,
        doc_service=doc_service,
        bm25_index=get_bm25_index(),
        ocr_engine=ocr_engine
    )

    try:
        user_id = get_jwt_identity()
        doc_info = doc_service.save_uploaded_file(file, user_id=user_id)
        
        pipeline.ingest_async(
            file_path=doc_info['file_path'],
            doc_id=doc_info['doc_id'],
            filename=doc_info['filename']
        )
        
        return jsonify({
            "message": "File uploaded successfully. Processing started.",
            "doc_id": doc_info['doc_id']
        }), 202

    except DuplicateDocumentError as de:
        # Upload dedup (RAG-011): file trùng nội dung document đã có -> 409, KHÔNG
        # lưu/ingest lại. Trả doc_id đã tồn tại để client dùng lại (idempotent).
        return jsonify({
            "error": str(de),
            "code": "duplicate_document",
            "existing_doc_id": de.existing_doc_id,
            "existing_name": de.existing_name,
        }), 409
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500

@documents_bp.route('/', methods=['GET'])
@jwt_required()
def list_documents():
    vector_ops = VectorStoreOps()
    doc_service = DocumentService(vector_ops, upload_folder=current_app.config['UPLOAD_FOLDER'])

    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    search = request.args.get('search', '')
    status = request.args.get('status', None)

    # Document-level ACL — luôn lấy từ JWT server-side (không phải query param).
    department = get_jwt().get("department")
    is_admin = get_jwt().get("role") == "admin"

    result = doc_service.get_documents(page=page, limit=limit, search=search, status=status,
                                        acl_department=department, acl_bypass=is_admin)
    return jsonify(result)


@documents_bp.route('/<doc_id>', methods=['GET'])
@jwt_required()
def get_document_status(doc_id):
    vector_ops = VectorStoreOps()
    doc_service = DocumentService(vector_ops, upload_folder=current_app.config['UPLOAD_FOLDER'])

    department = get_jwt().get("department")
    is_admin = get_jwt().get("role") == "admin"

    doc = doc_service.get_document_by_id(doc_id, acl_department=department, acl_bypass=is_admin)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    return jsonify(doc)


@documents_bp.route('/<doc_id>/metadata', methods=['PATCH'])
@jwt_required()
def update_document_metadata(doc_id):
    """Gán document_status/document_type/department cho document đã ingest"""
    vector_ops = VectorStoreOps()
    # bm25_index BẮT BUỘC ở đây: đổi ACL/metadata phải refresh BM25 để nhánh
    # keyword không lọc theo giá trị cũ (rò rỉ tài liệu) — xem
    # DocumentService.update_document_metadata().
    doc_service = DocumentService(
        vector_ops,
        upload_folder=current_app.config['UPLOAD_FOLDER'],
        bm25_index=get_bm25_index(),
    )

    doc = doc_service.get_document_by_id(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    is_admin = get_jwt().get("role") == "admin"

    # Cùng quy tắc phân quyền với xóa document: chủ sở hữu hoặc admin.
    if not is_admin and doc.get("uploaded_by") != get_jwt_identity():
        return jsonify({"error": "Bạn không có quyền sửa tài liệu này."}), 403

    data = request.json or {}
    document_status = data.get("document_status")
    document_type = data.get("document_type")
    department = data.get("department")

    if department is not None and not is_admin:
        return jsonify({"error": "Chỉ admin mới được gán phòng ban cho tài liệu."}), 403

    try:
        updated = doc_service.update_document_metadata(
            doc_id, document_status=document_status, document_type=document_type, department=department
        )
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    if not updated:
        return jsonify({"error": "Không có trường nào để cập nhật (cần document_status, document_type hoặc department)."}), 400

    return jsonify({"message": f"Đã cập nhật metadata cho document {doc_id}."})


@documents_bp.route('/<doc_id>', methods=['DELETE'])
@jwt_required()
def delete_document(doc_id):
    vector_ops = VectorStoreOps()
    doc_service = DocumentService(
        vector_ops,
        upload_folder=current_app.config['UPLOAD_FOLDER'],
        bm25_index=get_bm25_index()
    )

    doc = doc_service.get_document_by_id(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    # Nhân viên chỉ xóa được tài liệu do chính mình upload; admin xóa được bất kỳ.
    if get_jwt().get("role") != "admin" and doc.get("uploaded_by") != get_jwt_identity():
        return jsonify({"error": "Bạn không có quyền xóa tài liệu này."}), 403

    success = doc_service.delete_document(doc_id)
    if not success:
        return jsonify({"error": "Document not found"}), 404

    return jsonify({"message": f"Document {doc_id} deleted successfully."})
