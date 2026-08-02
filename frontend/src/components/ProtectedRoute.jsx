import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function ProtectedRoute({ children, role }) {
  const { user, loading } = useAuth();

  if (loading) {
    return <div className="flex h-screen items-center justify-center text-slate">Đang tải...</div>;
  }
  if (!user) return <Navigate to="/login" replace />;
  if (role && user.role !== role) {
    return (
      <div className="flex h-screen items-center justify-center text-slate">
        Bạn không có quyền truy cập trang này.
      </div>
    );
  }
  return children;
}
