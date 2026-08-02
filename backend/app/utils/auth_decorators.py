from functools import wraps

from flask_jwt_extended import verify_jwt_in_request, get_jwt

from .response_utils import error_response


def roles_required(*allowed_roles):
    """Decorator RBAC — yêu cầu JWT hợp lệ VÀ claim 'role' nằm trong allowed_roles."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            role = get_jwt().get("role")
            if role not in allowed_roles:
                return error_response("Không có quyền truy cập.", status_code=403)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def admin_required(fn):
    return roles_required("admin")(fn)
