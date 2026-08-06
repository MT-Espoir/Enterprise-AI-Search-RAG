import logging
import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from ..models.chat_session import ChatSession, Message, SourceRef
from ..extensions import mongo
from bson import ObjectId

from ..core.retrieval import build_retriever
from ..core.retrieval import get_bm25_index
from ..core.query import QueryProcessor
from ..core.ranking import Reranker
from ..core.generation import Generator
from ..core.rag_pipeline import RAGPipeline
from ..core.guardrails import check_input
from ..vectorstore.operations import VectorStoreOps
from ..extensions import limiter, user_or_ip_key

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')

from ..ingestion.embedder.local_embedder import LocalEmbedder
from ..core.generation import LocalGenerator
from ..core.generation import ClaudeGenerator

def get_rag_pipeline():
    """Khởi tạo RAG Pipeline từ app config."""
    vector_ops = VectorStoreOps()
    api_key = current_app.config['GOOGLE_API_KEY']
    
    provider = current_app.config.get('LLM_PROVIDER', 'local')

    # LUÔN SỬ DỤNG LOCAL EMBEDDER (BGE-M3, offline, khớp với embedder dùng lúc ingest).
    # get_instance(): singleton, tránh nạp lại model mỗi request (xem LocalEmbedder).
    embedder = LocalEmbedder.get_instance(model_name=current_app.config.get('LOCAL_EMBEDDING_MODEL'))

    if provider == 'claude':
        generator = ClaudeGenerator(
            api_key=current_app.config.get('ANTHROPIC_API_KEY'),
            model_name=current_app.config.get('ANTHROPIC_MODEL'),
        )
    elif provider == 'gemini':
        generator = Generator(api_key=api_key)
    else:  # "local" (mặc định)
        generator = LocalGenerator(
            base_url=current_app.config.get('OLLAMA_BASE_URL'),
            model_name=current_app.config.get('LOCAL_LLM_MODEL')
        )
        
    strategy = current_app.config.get('RETRIEVAL_STRATEGY', 'hybrid')
    bm25_index = get_bm25_index() if strategy == "hybrid" else None
    retriever = build_retriever(
        strategy,
        ops=vector_ops,
        embedder=embedder,
        bm25_index=bm25_index,
        vector_pool_size=current_app.config.get('HYBRID_VECTOR_POOL_SIZE', 20),
        bm25_pool_size=current_app.config.get('HYBRID_BM25_POOL_SIZE', 20),
        rrf_k=current_app.config.get('HYBRID_RRF_K', 60),
        top_k=10,
    )

    query_processor = QueryProcessor(
        base_url=current_app.config.get('OLLAMA_BASE_URL'),
        model_name=current_app.config.get('LOCAL_LLM_MODEL'),
        default_strategy=strategy,
        default_top_k=10,
        slm_enabled=current_app.config.get('QUERY_PROCESSOR_SLM_ENABLED', True),
    )

    reranker = Reranker.get_instance(top_k=3)

    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        query_processor=query_processor,
        observability_enabled=current_app.config.get('OBSERVABILITY_ENABLED', True),
        decomposition_enabled=current_app.config.get('DECOMPOSITION_ENABLED', True),
        retrieval_guardrails_enabled=current_app.config.get('RETRIEVAL_GUARDRAILS_ENABLED', True),
        output_guardrails_enabled=current_app.config.get('OUTPUT_GUARDRAILS_ENABLED', True),
    )

@chat_bp.route('/message', methods=['POST'])
@limiter.limit(lambda: current_app.config.get("RATELIMIT_CHAT", "20 per minute"), key_func=user_or_ip_key)
@jwt_required()
def send_message():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    question = data['message']
    session_id = data.get('session_id')
    doc_id = data.get('doc_id') # Lọc theo 1 tài liệu cụ thể (optional)

    # Input Guardrail (Tier 1, Layer 1) — PHẢI chạy TRƯỚC mọi persist (session
    # tạo mới/session.add_message bên dưới), để câu hỏi bị từ chối không bao
    # giờ được ghi vào Mongo.
    if current_app.config.get('INPUT_GUARDRAILS_ENABLED', True):
        input_check = check_input(question)
        if not input_check.allowed:
            return jsonify({"error": f"Câu hỏi bị từ chối bởi guardrail: {input_check.reason}"}), 400
        if input_check.pii_detected:
            logger.warning(f"Input guardrail: phát hiện PII {input_check.pii_detected} trong câu hỏi (không chặn).")

    # Metadata filtering đa chiều (Phase 4 mục 4, xem rag_core_quality_roadmap.md
    # mục 6l) — optional, KHÔNG tự động áp dụng (vd không tự loại "hết hiệu lực")
    # để tránh âm thầm giấu kết quả; user/frontend chủ động chọn khi cần.
    filters = {}
    if data.get('document_status'):
        filters['document_status'] = data['document_status']
    if data.get('document_type'):
        filters['document_type'] = data['document_type']

    user_id = get_jwt_identity()
    is_admin = get_jwt().get("role") == "admin"
    # Document-level ACL (phòng ban) — LUÔN lấy từ JWT server-side, KHÔNG BAO
    # GIỜ đọc từ request.json, để client không thể tự spoof/mở rộng quyền truy
    # cập tài liệu qua body. admin bypass hoàn toàn (acl_bypass=True).
    acl_department = get_jwt().get("department")

    # Lấy hoặc tạo ChatSession — nhân viên chỉ gửi được tin nhắn vào session của
    # chính mình (admin không bị giới hạn), tránh nhắn nhầm vào session người khác.
    session = None
    if session_id:
        try:
            query = {"_id": ObjectId(session_id)}
            if not is_admin:
                query["user_id"] = user_id
            session_data = mongo.db[ChatSession.COLLECTION].find_one(query)
            if session_data:
                session = ChatSession.from_dict(session_data)
        except Exception:
            pass
            
    if not session:
        session = ChatSession(user_id=user_id, title=question[:30])
        result = mongo.db[ChatSession.COLLECTION].insert_one(session.to_dict())
        session._id = result.inserted_id

    # Lấy lịch sử chat (để LLM nhớ ngữ cảnh)
    # Lấy 6 tin nhắn gần nhất (3 cặp hỏi đáp)
    recent_messages = session.messages[-6:] if session.messages else []
    history = [{"role": msg.role, "content": msg.content} for msg in recent_messages]

    # Lưu câu hỏi của User
    session.add_message(role="user", content=question)
    
    # Chạy RAG Pipeline
    rag = get_rag_pipeline()
    request_id = str(uuid.uuid4())
    try:
        rag_result = rag.run(question=question, doc_id=doc_id, history=history, request_id=request_id, filters=filters or None,
                             acl_department=acl_department, acl_bypass=is_admin)
        
        # Lưu câu trả lời của Bot
        sources_ref = []
        for s in rag_result["sources"]:
            sources_ref.append(
                SourceRef(
                    doc_id=s["doc_id"],
                    filename=s["filename"],
                    chunk_text="", # Không lưu lại chunk_text để tiết kiệm DB
                    score=0.0, 
                    page=s.get("page")
                )
            )
            
        session.add_message(
            role="assistant",
            content=rag_result["answer"],
            sources=sources_ref,
            tokens_used=rag_result["tokens_used"],
            guardrail=rag_result.get("guardrail")
        )

        # Cập nhật Session vào DB
        mongo.db[ChatSession.COLLECTION].replace_one(
            {"_id": session._id},
            session.to_dict()
        )

        return jsonify({
            "session_id": str(session._id),
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "tokens_used": rag_result["tokens_used"],
            "guardrail": rag_result.get("guardrail")
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"LLM/RAG Error: {e}"}), 500

@chat_bp.route('/sessions', methods=['GET'])
@jwt_required()
def list_sessions():
    user_id = get_jwt_identity()
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    skip = (page - 1) * limit

    cursor = (
        mongo.db[ChatSession.COLLECTION]
        .find({"user_id": user_id}, {"messages": 0}) # Tránh load toàn bộ message
        .sort("updated_at", -1)
        .skip(skip)
        .limit(limit)
    )

    sessions = []
    for s in cursor:
        s["_id"] = str(s["_id"])
        sessions.append(s)

    total = mongo.db[ChatSession.COLLECTION].count_documents({"user_id": user_id})

    return jsonify({
        "sessions": sessions,
        "total": total,
        "page": page,
        "limit": limit
    })

@chat_bp.route('/sessions/<session_id>', methods=['GET'])
@jwt_required()
def get_session(session_id):
    try:
        query = {"_id": ObjectId(session_id)}
        if get_jwt().get("role") != "admin":
            query["user_id"] = get_jwt_identity()

        session_data = mongo.db[ChatSession.COLLECTION].find_one(query)
        if not session_data:
            return jsonify({"error": "Session not found"}), 404

        session_data["_id"] = str(session_data["_id"])
        return jsonify(session_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

