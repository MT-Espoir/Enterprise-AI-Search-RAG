import { createContext, useContext, useEffect, useState } from "react";
import { apiJson, setTokens, clearTokens, getAccessToken } from "../api/client";

const AuthContext = createContext(null);
const USER_KEY = "rag_user";

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Có token nhưng chưa rõ user (vd token còn hạn nhưng localStorage user bị xoá
    // thủ công) -> xác nhận lại qua /me. Không có token -> coi như chưa đăng nhập.
    async function restore() {
      if (getAccessToken() && !user) {
        try {
          const body = await apiJson("/api/auth/me");
          setUser(body.data);
          localStorage.setItem(USER_KEY, JSON.stringify(body.data));
        } catch {
          clearTokens();
          localStorage.removeItem(USER_KEY);
        }
      }
      setLoading(false);
    }
    restore();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function login(email, password) {
    const body = await apiJson("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    setTokens(body.data);
    setUser(body.data.user);
    localStorage.setItem(USER_KEY, JSON.stringify(body.data.user));
  }

  function logout() {
    clearTokens();
    localStorage.removeItem(USER_KEY);
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, logout, isAdmin: user?.role === "admin" }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth phải dùng trong AuthProvider");
  return ctx;
}
