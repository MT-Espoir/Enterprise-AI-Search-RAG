from flask import Blueprint, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt, create_access_token
from ..services.auth_service import AuthService, AuthError
from ..utils.response_utils import success_response, error_response
from ..extensions import limiter

auth_bp = Blueprint("auth", __name__)
# AuthService KHÔNG được khởi tạo ở module level: mongo.db còn là None lúc
# module này được import (trước khi create_app() gọi mongo.init_app(app)),
# nên self.db sẽ bị gán cứng None vĩnh viễn nếu khởi tạo ở đây. Khởi tạo mới
# trong từng route, đúng pattern documents.py/chat.py/collections.py đã dùng.

@auth_bp.route("/login", methods=["POST"])
@limiter.limit(lambda: current_app.config.get("RATELIMIT_LOGIN", "5 per minute"))
def login():
    """API Đăng nhập.

    Rate-limit chặt (mặc định 5/phút/IP) để chống brute-force đoán mật khẩu —
    đây là endpoint không cần JWT nên IP là khóa đếm hợp lý duy nhất. Vượt giới
    hạn -> 429 (xem create_app::ratelimit_handler)."""
    data = request.get_json()

    if not data or not data.get("email") or not data.get("password"):
        return error_response("Vui lòng cung cấp email và password.")

    try:
        result = AuthService().login(data["email"], data["password"])
        return success_response(data=result, message="Đăng nhập thành công")
    except AuthError as e:
        return error_response(e.message, status_code=e.status_code)

@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    """Lấy thông tin người dùng hiện tại (Cần có JWT Access Token)"""
    # Lấy _id (được lưu bên trong Access Token lúc login)
    current_user_id = get_jwt_identity()

    user = AuthService().get_user_by_id(current_user_id)
    if not user:
        return error_response("Không tìm thấy user", status_code=404)
        
    return success_response(data=user)

@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    """Xin cấp lại Access Token mới (Cần có JWT Refresh Token)"""
    current_user_id = get_jwt_identity()
    # role/department lấy từ claim sẵn có trong refresh token — giữ nguyên khi
    # cấp access token mới. THIẾU department ở đây sẽ làm rớt claim sau lần
    # refresh đầu tiên -> user bất ngờ chỉ còn thấy tài liệu chưa gán phòng ban
    # (đã phát hiện khi thiết kế Document ACL, xem production_target_architecture.md).
    role = get_jwt().get("role", "user")
    department = get_jwt().get("department", "")

    # Tạo Access token mới
    new_access_token = create_access_token(identity=current_user_id, additional_claims={"role": role, "department": department})
    
    return success_response(data={"access_token": new_access_token})
