import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await login(email, password);
      navigate("/chat");
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-ice-50">
      <form
        className="w-[360px] rounded-xl border border-ice-200 bg-white p-8 shadow-sm"
        onSubmit={handleSubmit}
      >
        <h1 className="text-xl font-semibold text-navy">Đăng nhập</h1>
        <p className="mt-1 text-sm text-slate">
          Tài khoản do quản trị viên cấp — không có đăng ký công khai.
        </p>

        <label className="mt-5 block text-sm font-medium text-navy">
          Email
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="mt-1 w-full rounded-lg border border-ice-200 px-3 py-2 text-sm text-navy focus:border-navy focus:outline-none focus:ring-1 focus:ring-navy"
          />
        </label>

        <label className="mt-4 block text-sm font-medium text-navy">
          Mật khẩu
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="mt-1 w-full rounded-lg border border-ice-200 px-3 py-2 text-sm text-navy focus:border-navy focus:outline-none focus:ring-1 focus:ring-navy"
          />
        </label>

        {error && <div className="mt-3 text-sm text-red-600">{error}</div>}

        <button
          type="submit"
          disabled={submitting}
          className="mt-6 w-full rounded-lg bg-navy py-2.5 text-sm font-semibold text-white transition-colors hover:bg-navy-light disabled:opacity-60"
        >
          {submitting ? "Đang đăng nhập..." : "Đăng nhập"}
        </button>
      </form>
    </div>
  );
}
