from ..models.user import User
from ..extensions import mongo
from flask_jwt_extended import create_access_token, create_refresh_token
from bson import ObjectId

class AuthError(Exception):
    """Custom exception để ném lỗi từ Service lên API"""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class AuthService:
    def __init__(self):
        # Trỏ đến database của MongoDB
        self.db = mongo.db

    def register(self, username, email, password) -> dict:
        """Đăng ký tài khoản mới"""
        # 1. Kiểm tra username hoặc email đã tồn tại chưa
        if self.db.users.find_one({"$or": [{"email": email}, {"username": username}]}):
            raise AuthError("Email hoặc Username đã được sử dụng.", 409)
            
        # 2. Tạo đối tượng User mới (Tầng Data)
        new_user = User(username=username, email=email)
        
        # 3. Yêu cầu Model tự băm mật khẩu
        new_user.set_password(password)
        
        # 4. Lưu vào MongoDB
        self.db.users.insert_one(new_user.to_db_dict())

        result = new_user.to_dict()
        result["_id"] = str(result["_id"])
        return result

    def login(self, email, password) -> dict:
        """Đăng nhập và cấp token"""
        # 1. Tìm user theo email
        user_data = self.db.users.find_one({"email": email})
        if not user_data:
            raise AuthError("Sai email hoặc mật khẩu.", 401)
            
        # 2. Khôi phục object User từ dictionary
        user = User.from_dict(user_data)
        
        # 3. Kiểm tra mật khẩu (hàm verify_password dùng bcrypt.checkpw)
        if not user.verify_password(password):
            raise AuthError("Sai email hoặc mật khẩu.", 401)
            
        # 4. Tạo JWT Token (identity là _id dạng chuỗi)
        # role nhúng sẵn vào claims để RBAC decorator không cần query Mongo mỗi request.
        # Giới hạn: nếu role user bị đổi sau khi đã có JWT, token cũ vẫn giữ role tại thời
        # điểm login tới khi hết hạn (mặc định 3600s) — hành vi chuẩn của JWT, chấp nhận được.
        user_id_str = str(user._id)
        claims = {"role": user.role, "department": user.department}
        access_token = create_access_token(identity=user_id_str, additional_claims=claims)
        refresh_token = create_refresh_token(identity=user_id_str, additional_claims=claims)
        
        user_dict = user.to_dict()
        user_dict["_id"] = str(user_dict["_id"])
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": user_dict
        }

    def get_user_by_id(self, user_id: str) -> dict:
        """Lấy thông tin người dùng qua ID"""
        user_data = self.db.users.find_one({"_id": ObjectId(user_id)})
        if not user_data:
            return None
        result = User.from_dict(user_data).to_dict()
        result["_id"] = str(result["_id"])
        return result
