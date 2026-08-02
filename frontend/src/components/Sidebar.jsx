import { Link, useLocation, useNavigate } from "react-router-dom";
import { MessageSquare, FileText, Users, LogOut, BarChart3 } from "lucide-react";
import { useAuth } from "../context/AuthContext";

const NAV_ITEMS = [
  { to: "/chat", label: "Hỏi đáp", icon: MessageSquare },
  { to: "/documents", label: "Tài liệu", icon: FileText },
  { to: "/admin/users", label: "Quản lý nhân viên", icon: Users, adminOnly: true },
  { to: "/admin/observability", label: "Dashboard hệ thống", icon: BarChart3, adminOnly: true },
];

function initials(name) {
  return (name || "?").trim().charAt(0).toUpperCase();
}

export default function Sidebar() {
  const { user, isAdmin, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <aside className="flex h-screen w-64 shrink-0 flex-col border-r border-ice-200 bg-white">
      {/* Profile — avatar dùng initials thật từ username, chức danh/phòng ban là
          placeholder tĩnh vì User model chưa có 2 field này. */}
      <div className="flex flex-col items-center gap-1.5 px-6 py-8 text-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-full border-2 border-navy bg-ice-100 text-xl font-semibold text-navy">
          {initials(user?.username)}
        </div>
        <div className="mt-1 text-base font-semibold text-navy">{user?.username}</div>
        <div className="text-xs italic text-slate/60">Chức danh: chưa cập nhật</div>
        <div className="text-xs italic text-slate/60">Phòng ban: chưa cập nhật</div>
      </div>

      <div className="mx-6 border-t border-ice-200" />

      <nav className="flex-1 space-y-1 px-3 py-6">
        {NAV_ITEMS.filter((item) => !item.adminOnly || isAdmin).map((item) => {
          const Icon = item.icon;
          const active = location.pathname.startsWith(item.to);
          return (
            <Link
              key={item.to}
              to={item.to}
              className={`flex items-center gap-3 rounded-lg border-l-[3px] px-3 py-2.5 text-sm font-medium transition-colors ${
                active
                  ? "border-navy bg-ice-100 text-navy"
                  : "border-transparent text-slate hover:bg-ice-50 hover:text-navy"
              }`}
            >
              <Icon size={18} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="mx-6 border-t border-ice-200" />

      <div className="px-3 py-4">
        <button
          onClick={() => {
            logout();
            navigate("/login");
          }}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-slate transition-colors hover:bg-ice-50 hover:text-navy"
        >
          <LogOut size={18} />
          Đăng xuất
        </button>
      </div>
    </aside>
  );
}
