import { useState } from "react";
import { Paperclip, LayoutTemplate, Smile, Send } from "lucide-react";
import { apiJson } from "../api/client";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  async function handleSend(e) {
    e.preventDefault();
    if (!input.trim()) return;

    const question = input;
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setSending(true);
    setError("");

    try {
      const body = await apiJson("/api/chat/message", {
        method: "POST",
        body: JSON.stringify({ message: question, session_id: sessionId }),
      });
      setSessionId(body.session_id);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: body.answer, sources: body.sources, guardrail: body.guardrail },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col px-6 py-6">
      <div className="flex-1 space-y-4 overflow-y-auto">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm ${
                m.role === "user"
                  ? "bg-navy text-white"
                  : "border border-ice-200 bg-white text-navy"
              }`}
            >
              <div>{m.content}</div>
              {m.sources?.length > 0 && (
                <div className="mt-2 text-xs opacity-70">
                  Nguồn: {m.sources.map((s) => s.filename).join(", ")}
                </div>
              )}
              {m.guardrail?.citation_warning && (
                <div
                  className="mt-1 text-xs text-amber-600"
                  title={`Chưa xác minh: ${(m.guardrail.unverified_citations || []).join(", ")}`}
                >
                  ⚠ Một số trích dẫn (Điều luật) trong câu trả lời chưa được xác minh trong tài liệu nguồn
                </div>
              )}
            </div>
          </div>
        ))}
        {sending && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-2xl border border-ice-200 bg-white px-4 py-3 text-sm text-slate">
              Đang trả lời...
            </div>
          </div>
        )}
      </div>

      {error && <div className="mt-2 text-sm text-red-600">{error}</div>}

      <form
        onSubmit={handleSend}
        className="mt-4 flex items-center gap-2 rounded-2xl border border-ice-200 bg-white px-3 py-2 shadow-sm"
      >
        {/* Tiện ích phụ — placeholder tĩnh, chưa có backend đính kèm/prompt template/emoji */}
        <button type="button" disabled title="Sắp ra mắt" className="cursor-not-allowed p-1.5 text-slate opacity-40">
          <Paperclip size={18} />
        </button>
        <button type="button" disabled title="Sắp ra mắt" className="cursor-not-allowed p-1.5 text-slate opacity-40">
          <LayoutTemplate size={18} />
        </button>
        <button type="button" disabled title="Sắp ra mắt" className="cursor-not-allowed p-1.5 text-slate opacity-40">
          <Smile size={18} />
        </button>

        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Hỏi về tài liệu..."
          disabled={sending}
          className="flex-1 border-none px-2 py-1.5 text-sm text-navy outline-none placeholder:text-slate/50"
        />

        <button
          type="submit"
          disabled={sending}
          className="flex items-center gap-1.5 rounded-xl bg-navy px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-navy-light disabled:opacity-60"
        >
          <Send size={16} />
          Gửi
        </button>
      </form>
    </div>
  );
}
