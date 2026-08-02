import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { apiJson } from "../api/client";

const LATENCY_LINES = [
  { key: "avg_total_ms", label: "Tổng", color: "#0f2a4a" },
  { key: "avg_generation_ms", label: "Generation", color: "#2563eb" },
  { key: "avg_retrieval_ms", label: "Retrieval", color: "#0ea5e9" },
  { key: "avg_rerank_ms", label: "Rerank", color: "#f59e0b" },
  { key: "avg_preprocessing_ms", label: "Preprocessing", color: "#94a3b8" },
];

function StatCard({ label, value, sub, alert }) {
  return (
    <div className="rounded-xl border border-ice-200 bg-white p-5">
      <div className="text-xs font-semibold uppercase tracking-wide text-slate">{label}</div>
      <div className={`mt-2 text-2xl font-bold ${alert ? "text-red-600" : "text-navy"}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-slate">{sub}</div>}
    </div>
  );
}

function formatBucketLabel(bucket) {
  // "2026-07-19T14:00:00Z" -> "19/07 14h"
  const d = new Date(bucket);
  if (Number.isNaN(d.getTime())) return bucket;
  return `${String(d.getDate()).padStart(2, "0")}/${String(d.getMonth() + 1).padStart(2, "0")} ${d.getHours()}h`;
}

export default function AdminObservability() {
  const [summary, setSummary] = useState(null);
  const [latencyBuckets, setLatencyBuckets] = useState([]);
  const [ocr, setOcr] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  async function loadAll() {
    setLoading(true);
    setError("");
    try {
      const [summaryBody, latencyBody, ocrBody] = await Promise.all([
        apiJson("/api/dashboard/summary"),
        apiJson("/api/dashboard/latency?days=7"),
        apiJson("/api/dashboard/ocr"),
      ]);
      setSummary(summaryBody.data);
      setLatencyBuckets(
        (latencyBody.data?.buckets || []).map((b) => ({ ...b, label: formatBucketLabel(b.bucket) }))
      );
      setOcr(ocrBody.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAll();
  }, []);

  const ocrFailPct =
    ocr?.overall_fail_rate !== null && ocr?.overall_fail_rate !== undefined
      ? `${Math.round(ocr.overall_fail_rate * 100)}%`
      : "—";
  const ocrAlert = ocr?.overall_fail_rate !== null && ocr?.overall_fail_rate > 0.1;

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-navy">Dashboard hệ thống</h1>
        <button
          onClick={loadAll}
          disabled={loading}
          className="rounded-lg border border-ice-200 px-3 py-1.5 text-sm font-medium text-navy hover:bg-ice-50 disabled:opacity-60"
        >
          {loading ? "Đang tải..." : "Làm mới"}
        </button>
      </div>

      {error && <div className="mt-3 text-sm text-red-600">{error}</div>}

      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard
          label="Tổng số request"
          value={summary?.total_requests ?? "—"}
          sub={`${summary?.requests_last_24h ?? 0} trong 24h qua`}
        />
        <StatCard
          label="Latency trung bình (24h)"
          value={summary?.avg_latency_ms_last_24h != null ? `${(summary.avg_latency_ms_last_24h / 1000).toFixed(1)}s` : "—"}
        />
        <StatCard
          label="Tỷ lệ OCR fail"
          value={ocrFailPct}
          sub={ocr ? `${ocr.total_pages_ocr_failed}/${ocr.total_pages_needing_ocr} trang` : ""}
          alert={ocrAlert}
        />
      </div>

      <div className="mt-6 rounded-xl border border-ice-200 bg-white p-6">
        <h2 className="text-sm font-semibold text-navy">Latency theo giờ (7 ngày gần nhất)</h2>
        {latencyBuckets.length === 0 ? (
          <p className="mt-4 text-sm text-slate">Chưa có dữ liệu request nào được ghi nhận.</p>
        ) : (
          <div className="mt-4 h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={latencyBuckets}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} unit="ms" />
                <Tooltip formatter={(v) => (v != null ? `${v} ms` : "—")} />
                <Legend />
                {LATENCY_LINES.map((line) => (
                  <Line
                    key={line.key}
                    type="monotone"
                    dataKey={line.key}
                    name={line.label}
                    stroke={line.color}
                    dot={false}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <div className="mt-6 rounded-xl border border-ice-200 bg-white p-6">
        <h2 className="text-sm font-semibold text-navy">OCR fail rate theo tài liệu</h2>
        {!ocr || ocr.per_document.length === 0 ? (
          <p className="mt-4 text-sm text-slate">Chưa có tài liệu nào từng qua OCR.</p>
        ) : (
          <div className="mt-4 overflow-hidden rounded-lg border border-ice-100">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-ice-200 bg-ice-50 text-left text-navy">
                  <th className="px-4 py-2 font-semibold">Tài liệu</th>
                  <th className="px-4 py-2 font-semibold">Trang cần OCR</th>
                  <th className="px-4 py-2 font-semibold">Trang OCR fail</th>
                  <th className="px-4 py-2 font-semibold">Tỷ lệ fail</th>
                </tr>
              </thead>
              <tbody>
                {ocr.per_document.map((d) => (
                  <tr key={d.doc_id} className="border-b border-ice-100 last:border-0">
                    <td className="px-4 py-2 text-navy">{d.filename}</td>
                    <td className="px-4 py-2 text-slate">{d.pages_needing_ocr}</td>
                    <td className="px-4 py-2 text-slate">{d.pages_ocr_failed}</td>
                    <td className="px-4 py-2">
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                          d.fail_rate > 0.1 ? "bg-red-100 text-red-700" : "bg-ice-100 text-navy"
                        }`}
                      >
                        {Math.round(d.fail_rate * 100)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
