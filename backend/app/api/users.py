from flask import Blueprint, request
from ..services.auth_service import AuthService, AuthError
from ..models.user import User
from ..models.department import DEPARTMENTS, DEPARTMENT_LABELS
from ..extensions import mongo
from ..utils.auth_decorators import admin_required
from ..utils.response_utils import success_response, error_response
from bson import ObjectId

users_bp = Blueprint("users", __name__)
# AuthService khởi tạo mới trong từng route (không phải module-level) — xem
# giải thích trong auth.py.


@users_bp.route("/", methods=["POST"])
@admin_required
def create_user():
    """Admin tạo tài khoản nhân viên mới. /register công khai đã bị đóng —
    đây là cách duy nhất tạo user mới (trừ script bootstrap admin đầu tiên)."""
    data = request.get_json()
    if not data or not data.get("email") or not data.get("password") or not data.get("username"):
        return error_response("Vui lòng cung cấp đầy đủ username, email và password.")

    role = data.get("role", "user")
    if role not in User.ROLES:
        return error_response(f"Role không hợp lệ. Chỉ chấp nhận: {User.ROLES}")

    department = data.get("department", "")
    if department and department not in DEPARTMENTS:
        return error_response(f"Department không hợp lệ. Chỉ chấp nhận: {DEPARTMENTS} hoặc rỗng.")

    try:
        user = AuthService().register(data["username"], data["email"], data["password"])
        # AuthService.register() luôn tạo role="user", department="" mặc định —
        # đổi thêm bằng update_one riêng nếu admin chỉ định khác (cùng pattern
        # đã có cho role, tránh đổi signature register()).
        if role != "user" or department:
            mongo.db[User.COLLECTION].update_one({"_id": ObjectId(user["_id"])}, {"$set": {"role": role, "department": department}})
            user["role"] = role
            user["department"] = department
        return success_response(data=user, message="Tạo tài khoản thành công", status_code=201)
    except AuthError as e:
        return error_response(e.message, status_code=e.status_code)


@users_bp.route("/departments", methods=["GET"])
@admin_required
def list_departments():
    """Danh sách phòng ban cố định (Document-level ACL) cho dropdown frontend —
    single source of truth, tránh trùng lặp label tiếng Việt ở phía client."""
    return success_response(data=[{"code": d, "label": DEPARTMENT_LABELS[d]} for d in DEPARTMENTS])


@users_bp.route("/", methods=["GET"])
@admin_required
def list_users():
    users = [User.from_dict(u).to_dict() for u in mongo.db[User.COLLECTION].find({})]
    for u in users:
        u["_id"] = str(u["_id"])
    return success_response(data=users)


@users_bp.route("/<user_id>/role", methods=["PATCH"])
@admin_required
def update_user_role(user_id):
    data = request.get_json()
    role = data.get("role") if data else None
    if role not in User.ROLES:
        return error_response(f"Role không hợp lệ. Chỉ chấp nhận: {User.ROLES}")

    result = mongo.db[User.COLLECTION].update_one({"_id": ObjectId(user_id)}, {"$set": {"role": role}})
    if result.matched_count == 0:
        return error_response("Không tìm thấy user.", status_code=404)

    return success_response(message=f"Đã đổi role thành '{role}'.")


@users_bp.route("/<user_id>/department", methods=["PATCH"])
@admin_required
def update_user_department(user_id):
    """Gán phòng ban cho user (Document-level ACL) — admin-only, giống hệt
    pattern update_user_role(). department="" hợp lệ (bỏ gán)."""
    data = request.get_json()
    department = data.get("department", "") if data else ""
    if department and department not in DEPARTMENTS:
        return error_response(f"Department không hợp lệ. Chỉ chấp nhận: {DEPARTMENTS} hoặc rỗng.")

    result = mongo.db[User.COLLECTION].update_one({"_id": ObjectId(user_id)}, {"$set": {"department": department}})
    if result.matched_count == 0:
        return error_response("Không tìm thấy user.", status_code=404)

    return success_response(message=f"Đã đổi department thành '{department or '(chưa gán)'}'.")
