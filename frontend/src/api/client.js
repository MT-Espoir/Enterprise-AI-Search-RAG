const TOKEN_KEY = "rag_access_token";
const REFRESH_KEY = "rag_refresh_token";

export function getAccessToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setTokens({ access_token, refresh_token }) {
  localStorage.setItem(TOKEN_KEY, access_token);
  if (refresh_token) localStorage.setItem(REFRESH_KEY, refresh_token);
}

export function clearTokens() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_KEY);
}

async function tryRefresh() {
  const refreshToken = localStorage.getItem(REFRESH_KEY);
  if (!refreshToken) return false;

  const res = await fetch("/api/auth/refresh", {
    method: "POST",
    headers: { Authorization: `Bearer ${refreshToken}` },
  });
  if (!res.ok) return false;

  const body = await res.json();
  localStorage.setItem(TOKEN_KEY, body.data.access_token);
  return true;
}

/**
 * Wrapper fetch: gắn sẵn Authorization header, tự thử refresh 1 lần khi gặp 401
 * trước khi trả lỗi thẳng cho caller (caller tự quyết định logout nếu vẫn 401).
 */
export async function apiFetch(path, options = {}) {
  const doFetch = () => {
    const token = getAccessToken();
    const headers = { ...(options.headers || {}) };
    if (token) headers.Authorization = `Bearer ${token}`;
    return fetch(path, { ...options, headers });
  };

  let res = await doFetch();
  if (res.status === 401) {
    const refreshed = await tryRefresh();
    if (refreshed) res = await doFetch();
  }
  return res;
}

export async function apiJson(path, options = {}) {
  const res = await apiFetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = body?.error?.message || body?.error || `Lỗi ${res.status}`;
    throw new Error(message);
  }
  return body;
}
