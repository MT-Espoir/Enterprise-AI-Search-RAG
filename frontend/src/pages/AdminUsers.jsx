import { useEffect, useState } from "react";
import { apiJson } from "../api/client";

export default function AdminUsers() {
  const [users, setUsers] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [error, setError] = useState("");
  const [form, setForm] = useState({ username: "", email: "", password: "", role: "user", department: "" });
  const [submitting, setSubmitting] = useState(false);

  async function loadUsers() {
    try {
      const body = await apiJson("/api/users/");
      setUsers(body.data || []);
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadDepartments() {
    try {
      const body = await apiJson("/api/users/departments");
      setDepartments(body.data || []);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    loadUsers();
    loadDepartments();
  }, []);

  async function handleCreate(e) {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      await apiJson("/api/users/", { method: "POST", body: JSON.stringify(form) });
      setForm({ username: "", email: "", password: "", role: "user", department: "" });
      await loadUsers();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleRoleChange(userId, role) {
    try {
      await apiJson(`/api/users/${userId}/role`, { method: "PATCH", body: JSON.stringify({ role }) });
      await loadUsers();
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleDepartmentChange(userId, department) {
    try {
      await apiJson(`/api/users/${userId}/department`, { method: "PATCH", body: JSON.stringify({ department }) });
      await loadUsers();
    } catch (err) {
      setError(err.message);
    }
  }

  const inputClass =
    "rounded-lg border border-ice-200 px-3 py-2 text-sm text-navy focus:border-navy focus:outline-none focus:ring-1 focus:ring-navy";

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <h1 className="text-xl font-semibold text-navy">Quản lý nhân viên</h1>

      <form
        onSubmit={handleCreate}
        className="mt-6 rounded-xl border border-ice-200 bg-white p-6"
      >
        <h2 className="text-sm font-semibold text-navy">Tạo tài khoản mới</h2>
        <div className="mt-4 flex flex-wrap items-end gap-3">
          <input
            placeholder="Username"
            value={form.username}
            onChange={(e) => setForm({ ...form, username: e.target.value })}
            required
            className={inputClass}
          />
          <input
            type="email"
            placeholder="Email"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
            required
            className={inputClass}
          />
          <input
            type="password"
            placeholder="Mật khẩu"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            required
            className={inputClass}
          />
          <select
            value={form.role}
            onChange={(e) => setForm({ ...form, role: e.target.value })}
            className={inputClass}
          >
            <option value="user">Nhân viên</option>
            <option value="admin">Admin</option>
          </select>
          <select
            value={form.department}
            onChange={(e) => setForm({ ...form, department: e.target.value })}
            className={inputClass}
          >
            <option value="">-- Chưa gán phòng ban --</option>
            {departments.map((d) => (
              <option key={d.code} value={d.code}>{d.label}</option>
            ))}
          </select>
          <button
            type="submit"
            disabled={submitting}
            className="rounded-lg bg-navy px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-navy-light disabled:opacity-60"
          >
            {submitting ? "Đang tạo..." : "Tạo tài khoản"}
          </button>
        </div>
      </form>

      {error && <div className="mt-3 text-sm text-red-600">{error}</div>}

      <div className="mt-6 overflow-hidden rounded-xl border border-ice-200 bg-white">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-ice-200 bg-ice-50 text-left text-navy">
              <th className="px-4 py-3 font-semibold">Username</th>
              <th className="px-4 py-3 font-semibold">Email</th>
              <th className="px-4 py-3 font-semibold">Role</th>
              <th className="px-4 py-3 font-semibold">Đổi role</th>
              <th className="px-4 py-3 font-semibold">Phòng ban</th>
              <th className="px-4 py-3 font-semibold">Đổi phòng ban</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u._id} className="border-b border-ice-100 last:border-0 hover:bg-ice-50">
                <td className="px-4 py-3 text-navy">{u.username}</td>
                <td className="px-4 py-3 text-slate">{u.email}</td>
                <td className="px-4 py-3">
                  <span
                    className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                      u.role === "admin" ? "bg-navy text-white" : "bg-ice-100 text-navy"
                    }`}
                  >
                    {u.role}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <select
                    value={u.role}
                    onChange={(e) => handleRoleChange(u._id, e.target.value)}
                    className="rounded-lg border border-ice-200 px-2 py-1 text-sm text-navy focus:border-navy focus:outline-none"
                  >
                    <option value="user">Nhân viên</option>
                    <option value="admin">Admin</option>
                  </select>
                </td>
                <td className="px-4 py-3 text-slate">
                  {departments.find((d) => d.code === u.department)?.label || "—"}
                </td>
                <td className="px-4 py-3">
                  <select
                    value={u.department || ""}
                    onChange={(e) => handleDepartmentChange(u._id, e.target.value)}
                    className="rounded-lg border border-ice-200 px-2 py-1 text-sm text-navy focus:border-navy focus:outline-none"
                  >
                    <option value="">-- Chưa gán --</option>
                    {departments.map((d) => (
                      <option key={d.code} value={d.code}>{d.label}</option>
                    ))}
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
