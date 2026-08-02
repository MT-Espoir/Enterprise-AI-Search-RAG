import { useEffect, useState } from "react";
import { Upload, Trash2 } from "lucide-react";
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
