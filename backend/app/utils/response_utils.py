from flask import jsonify

def success_response(data=None, message=None, meta=None, status_code=200):
    """Format chuẩn cho API trả về thành công"""
    response = {
        "success": True
    }
    if data is not None:
        response["data"] = data
    if message:
        response["message"] = message
    if meta:
        response["meta"] = meta
        
    return jsonify(response), status_code

def error_response(message: str, error_code: str = None, status_code: int = 400):
    """Format chuẩn cho API báo lỗi"""
    response = {
        "success": False,
        "error": {
            "message": message
        }
    }
    if error_code:
        response["error"]["code"] = error_code
        
    return jsonify(response), status_code

def paginated_response(items: list, total: int, page: int, limit: int, status_code=200):
    """Format cho API trả về danh sách có phân trang"""
    return success_response(
        data=items,
        meta={
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit if limit > 0 else 1
        },
        status_code=status_code
    )
