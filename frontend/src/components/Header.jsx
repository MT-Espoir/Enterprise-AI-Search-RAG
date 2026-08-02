import { Search, Bell } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function Header() {
  const { user } = useAuth();

  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-ice-200 bg-white px-6">
      {/* Global Search — placeholder tĩnh, chưa có endpoint tìm kiếm toàn hệ thống */}
      <div className="relative w-80">
        <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate/50" />
        <input
          type="text"
          disabled
          placeholder="Tìm kiếm toàn hệ thống (sắp ra mắt)"
          title="Chưa khả dụng"
          className="w-full cursor-not-allowed rounded-lg border border-ice-200 bg-ice-50 py-2 pl-9 pr-3 text-sm text-slate placeholder:text-slate/50"
        />
      </div>

      <div className="flex items-center gap-5">
        {/* Notification — placeholder tĩnh, chưa có backend notification */}
        <button type="button" disabled title="Chưa khả dụng" className="cursor-not-allowed text-slate opacity-40">
          <Bell size={20} />
        </button>

        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-navy">{user?.username}</span>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
              user?.role === "admin" ? "bg-navy text-white" : "bg-ice-100 text-navy"
            }`}
          >
            {user?.role === "admin" ? "Admin" : "Nhân viên"}
          </span>
        </div>
      </div>
    </header>
  );
}
