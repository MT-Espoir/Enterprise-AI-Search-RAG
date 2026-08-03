import { useEffect, useState } from "react";
import { Upload, Trash2, Search, Download, FileText } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { apiFetch, apiJson } from "../api/client";

const PAGE_SIZE = 20;

const STATUS_STYLE = {
  done: "bg-green-50 text-green-700",
  processing: "bg-ice-100 text-navy",
  pending: "bg-gray-100 text-slate",
  failed: "bg-red-50 text-red-700",
};

export default function Documents() {
  const { user, isAdmin } = useAuth();
  const [documents, setDocuments] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState(null); // null = chưa tìm; [] = tìm rồi, rỗng
  const [searching, setSearching] = useState(false);

  async function loadDocuments(targetPage = page) {
    setLoading(true);
    try {
      // documents.py trả jsonify thô (không qua success_response) -> body chính là { documents, total, page, pages, ... }
      const res = await apiFetch(`/api/documents/?page=${targetPage}&limit=${PAGE_SIZE}`);
      const body = await res.json();
      if (!res.ok) throw new Error(body?.error || "Không tải được danh sách tài liệu");
      setDocuments(body.documents || []);
      setTotal(body.total ?? 0);
      setTotalPages(body.pages ?? 1);
      setPage(targetPage);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadDepartments() {
    try {
      const body = await apiJson("/api/users/departments");
      setDepartments(body.data || []);
    } catch (err) {
      // Không phải admin -> 403, bỏ qua im lặng (chỉ admin mới cần dropdown này)
    }
  }

  useEffect(() => {
    loadDocuments(1);
    if (isAdmin) loadDepartments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleDepartmentChange(docId, department) {
    try {
      await apiJson(`/api/documents/${docId}/metadata`, { method: "PATCH", body: JSON.stringify({ department }) });
      await loadDocuments(page);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await apiFetch("/api/documents/upload", { method: "POST", body: formData });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.error || "Upload thất bại");
      await loadDocuments(1); // tài liệu mới nhất nằm ở trang 1 (sort theo created_at giảm dần)
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function handleSearch(e) {
    e?.preventDefault();
    const q = searchQuery.trim();
    if (!q) {
      setSearchResults(null);
      return;
    }
    setSearching(true);
    setError("");
    try {
      const res = await apiFetch(`/api/documents/search?q=${encodeURIComponent(q)}`);
      const body = await res.json();
      if (!res.ok) throw new Error(body?.error || "Tìm kiếm thất bại");
      setSearchResults(body.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  }

  async function handleDownload(docId, name) {
    // Download cần JWT header -> tải qua apiFetch thành blob rồi trigger tải xuống,
    // không dùng <a href> trực tiếp (sẽ thiếu Authorization -> 401).
    try {
      const res = await apiFetch(`/api/documents/${docId}/download`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Tải file thất bại");
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name || "document";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleDelete(docId) {
    if (!confirm("Xóa tài liệu này?")) return;
    try {
      const res = await apiFetch(`/api/documents/${docId}`, { method: "DELETE" });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.error || "Xóa thất bại");
      // Xóa hết item cuối cùng của trang cuối -> lùi về trang trước để tránh trang trống
      const targetPage = documents.length === 1 && page > 1 ? page - 1 : page;
      await loadDocuments(targetPage);
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-navy">Tài liệu</h1>
        <label className="flex cursor-pointer items-center gap-2 rounded-lg bg-navy px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-navy-light">
          <Upload size={16} />
          {uploading ? "Đang tải lên..." : "Tải lên tài liệu"}
          <input type="file" hidden onChange={handleUpload} disabled={uploading} />
        </label>
      </div>

      {error && <div className="mt-3 text-sm text-red-600">{error}</div>}

      {/* Tìm kiếm file để dẫn hướng + tải về (tách khỏi RAG hỏi-đáp; lọc theo ACL phòng ban) */}
      <form onSubmit={handleSearch} className="mt-5 flex gap-2">
        <div className="relative flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate" />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Tìm file / biểu mẫu (vd: biểu mẫu hợp đồng lao động)"
            className="w-full rounded-lg border border-ice-200 py-2 pl-9 pr-3 text-sm text-navy focus:border-navy focus:outline-none"
          />
        </div>
        <button
          type="submit"
          disabled={searching}
          className="rounded-lg bg-navy px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-navy-light disabled:opacity-50"
        >
          {searching ? "Đang tìm..." : "Tìm file"}
        </button>
        {searchResults !== null && (
          <button
            type="button"
            onClick={() => {
              setSearchResults(null);
              setSearchQuery("");
            }}
            className="rounded-lg border border-ice-200 px-3 py-2 text-sm text-slate hover:bg-ice-50"
          >
            Xóa
          </button>
        )}
      </form>

      {searchResults !== null && (
        <div className="mt-4 rounded-xl border border-ice-200 bg-white p-4">
          <div className="mb-2 text-sm font-semibold text-navy">
            Kết quả tìm kiếm ({searchResults.length})
          </div>
          {searchResults.length === 0 ? (
            <div className="text-sm text-slate">
              Không tìm thấy file phù hợp (hoặc file không thuộc phòng ban của bạn).
            </div>
          ) : (
            <ul className="divide-y divide-ice-100">
              {searchResults.map((r) => (
                <li key={r.doc_id} className="flex items-center justify-between py-2.5">
                  <div className="flex items-center gap-2">
                    <FileText size={16} className="shrink-0 text-navy" />
                    <div>
                      <div className="text-sm text-navy">{r.original_name}</div>
                      <div className="text-xs text-slate">
                        {r.match_reason === "filename" ? "khớp tên file" : "khớp nội dung"}
                        {r.document_type ? ` · ${r.document_type}` : ""}
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDownload(r.doc_id, r.original_name)}
                    className="inline-flex shrink-0 items-center gap-1 rounded-lg border border-navy px-3 py-1.5 text-sm font-medium text-navy hover:bg-ice-50"
                  >
                    <Download size={14} /> Tải về
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {loading ? (
        <div className="mt-6 text-sm text-slate">Đang tải...</div>
      ) : (
        <div className="mt-6 overflow-hidden rounded-xl border border-ice-200 bg-white">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-ice-200 bg-ice-50 text-left text-navy">
                <th className="px-4 py-3 font-semibold">Tên file</th>
                <th className="px-4 py-3 font-semibold">Trạng thái</th>
                <th className="px-4 py-3 font-semibold">Số chunk</th>
                <th className="px-4 py-3 font-semibold">Phòng ban</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => {
                const canDelete = isAdmin || doc.uploaded_by === user?._id;
                return (
                  <tr key={doc._id} className="border-b border-ice-100 last:border-0 hover:bg-ice-50">
                    <td className="px-4 py-3 text-navy">{doc.original_name}</td>
                    <td className="px-4 py-3">
                      <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${STATUS_STYLE[doc.status] || "bg-gray-100 text-slate"}`}>
                        {doc.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate">{doc.chunk_count ?? "-"}</td>
                    <td className="px-4 py-3">
                      {isAdmin ? (
                        <select
                          value={doc.department || ""}
                          onChange={(e) => handleDepartmentChange(doc._id, e.target.value)}
                          className="rounded-lg border border-ice-200 px-2 py-1 text-sm text-navy focus:border-navy focus:outline-none"
                        >
                          <option value="">-- Chưa gán --</option>
                          {departments.map((d) => (
                            <option key={d.code} value={d.code}>{d.label}</option>
                          ))}
                        </select>
                      ) : (
                        <span className="text-slate">
                          {departments.find((d) => d.code === doc.department)?.label || doc.department || "—"}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {canDelete && (
                        <button
                          onClick={() => handleDelete(doc._id)}
                          className="inline-flex items-center gap-1 text-sm font-medium text-red-600 hover:text-red-700"
                        >
                          <Trash2 size={14} />
                          Xóa
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {!loading && totalPages > 1 && (
        <div className="mt-4 flex items-center gap-4 text-sm">
          <button
            disabled={page <= 1}
            onClick={() => loadDocuments(page - 1)}
            className="rounded-lg border border-ice-200 bg-white px-3 py-1.5 text-navy disabled:opacity-40"
          >
            ← Trước
          </button>
          <span className="text-slate">
            Trang {page}/{totalPages} ({total} tài liệu)
          </span>
          <button
            disabled={page >= totalPages}
            onClick={() => loadDocuments(page + 1)}
            className="rounded-lg border border-ice-200 bg-white px-3 py-1.5 text-navy disabled:opacity-40"
          >
            Sau →
          </button>
        </div>
      )}
    </div>
  );
}
