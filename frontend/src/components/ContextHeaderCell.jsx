import { useState } from "react";
import { Check, Pencil, RefreshCw, Undo2, X } from "lucide-react";
import { apiJson } from "../api/client";

// Header ngữ cảnh tài liệu (backend: app/services/context_header_service.py). Header HIỆU LỰC =
// bản đã duyệt nếu có, ngược lại bản hệ thống gợi ý. Đổi header chỉ cập nhật MongoDB — vector của
// tài liệu được tạo lại sau bởi job embed lại, nên hiện nhãn "Chờ cập nhật vector" khi 2 version lệch.
const MAX_HEADER_CHARS = 250;

const BADGE = {
  approved: { text: "Đã duyệt", cls: "bg-green-50 text-green-700" },
  review: { text: "Cần duyệt", cls: "bg-amber-50 text-amber-700" },
  suggested: { text: "Gợi ý", cls: "bg-ice-100 text-navy" },
};

// Trường của 1 document trong danh sách ← state trả về từ API context-header
export function headerStateToDocFields(state) {
  return {
    context_header: state.approved,
    context_header_suggested: state.suggested,
    context_header_suggested_source: state.suggested_source,
    context_header_source: state.source,
    context_header_needs_review: state.needs_review,
    context_header_version: state.version,
    embedded_header_version: state.embedded_version,
  };
}

function IconButton({ title, onClick, disabled, children }) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium text-navy hover:bg-ice-100 disabled:opacity-40"
    >
      {children}
    </button>
  );
}

export default function ContextHeaderCell({ doc, canEdit, onChanged, onError }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);

  const approved = doc.context_header;
  const suggested = doc.context_header_suggested;
  const effective = approved || suggested;
  const status = approved ? "approved" : suggested ? (doc.context_header_needs_review ? "review" : "suggested") : "none";
  const needsReembed = (doc.context_header_version || 0) !== (doc.embedded_header_version || 0);
  const ready = doc.status === "done";

  async function call(path, options) {
    setBusy(true);
    try {
      const state = await apiJson(`/api/documents/${doc._id}/context-header${path}`, options);
      onChanged(doc._id, headerStateToDocFields(state));
      return true;
    } catch (err) {
      onError(err.message);
      return false;
    } finally {
      setBusy(false);
    }
  }

  const approve = () => call("", { method: "PATCH", body: JSON.stringify({ approve: true }) });
  const clearApproved = () => call("", { method: "PATCH", body: JSON.stringify({ context_header: null }) });
  // Gợi ý lại có thể gọi SLM local (vài chục giây trên CPU) — nút quay trong lúc chờ
  const resuggest = () => call("/suggest", { method: "POST" });

  async function save() {
    if (await call("", { method: "PATCH", body: JSON.stringify({ context_header: draft }) })) setEditing(false);
  }

  if (editing) {
    return (
      <div className="max-w-xl">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          maxLength={MAX_HEADER_CHARS}
          rows={3}
          autoFocus
          placeholder="[Loại tài liệu, số hiệu / năm, đơn vị ban hành]"
          className="w-full rounded-lg border border-ice-200 px-2 py-1 text-xs text-navy focus:border-navy focus:outline-none"
        />
        <div className="mt-1 flex items-center justify-between">
          <span className="text-[11px] text-slate">
            {draft.length}/{MAX_HEADER_CHARS}
          </span>
          <div className="flex gap-1">
            <IconButton title="Lưu" onClick={save} disabled={busy || !draft.trim()}>
              <Check size={13} /> Lưu
            </IconButton>
            <IconButton title="Hủy" onClick={() => setEditing(false)} disabled={busy}>
              <X size={13} /> Hủy
            </IconButton>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-xl">
      {effective ? (
        <div className={`line-clamp-2 text-xs ${approved ? "text-navy" : "text-slate"}`} title={effective}>
          {effective}
        </div>
      ) : (
        <div className="text-xs text-slate">{ready ? "Chưa có header" : "Chờ xử lý xong"}</div>
      )}

      <div className="mt-1 flex flex-wrap items-center gap-1">
        {BADGE[status] && (
          <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${BADGE[status].cls}`}>
            {BADGE[status].text}
          </span>
        )}
        {needsReembed && (
          <span
            className="rounded-full bg-gray-100 px-2 py-0.5 text-[11px] font-medium text-slate"
            title="Header đã đổi nhưng vector tìm kiếm của tài liệu chưa được tạo lại"
          >
            Chờ cập nhật vector
          </span>
        )}

        {canEdit && ready && (
          <>
            {status !== "approved" && suggested && (
              <IconButton title="Duyệt nguyên văn header gợi ý" onClick={approve} disabled={busy}>
                <Check size={13} /> Duyệt
              </IconButton>
            )}
            <IconButton
              title="Sửa header"
              onClick={() => {
                setDraft(effective || "");
                setEditing(true);
              }}
              disabled={busy}
            >
              <Pencil size={13} /> Sửa
            </IconButton>
            <IconButton title="Gợi ý lại từ nội dung tài liệu" onClick={resuggest} disabled={busy}>
              <RefreshCw size={13} className={busy ? "animate-spin" : ""} /> Gợi ý lại
            </IconButton>
            {approved && (
              <IconButton title="Bỏ bản đã duyệt, quay về header gợi ý" onClick={clearApproved} disabled={busy}>
                <Undo2 size={13} /> Bỏ duyệt
              </IconButton>
            )}
          </>
        )}
      </div>
    </div>
  );
}
