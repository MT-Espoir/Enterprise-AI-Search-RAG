"""
batch_upload_documents.py — Upload HÀNG LOẠT tài liệu qua API app đang chạy, rồi
sinh BÁO CÁO VERIFY per-file (đúng luồng production: dedup SHA-256 + OCR + ocr_stats
+ ACL — không re-wire pipeline, không bỏ qua bước nào).

Yêu cầu: chỉ dùng `requests` (đã có sẵn trong project).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

import requests

# Đồng bộ với IngestionPipeline.PARSER_MAP — chỉ upload các đuôi thật sự parse được.
SUPPORTED_EXT = {".pdf", ".docx", ".md", ".txt", ".xlsx", ".csv"}

# Ngưỡng gắn cờ NGHI NGỜ (không chặn, chỉ để soi tay):
_SUSPECT_MIN_CHUNKS = 2       # done mà < 2 chunk → khả năng rỗng/parse hỏng
_SUSPECT_OCR_FAIL_RATE = 0.10  # >10% trang OCR thất bại → chất lượng scan kém


class Auth:
    """Giữ token + TỰ ĐĂNG NHẬP LẠI khi token hết hạn (401). Access token mặc định
    sống 1 giờ (JWT_ACCESS_TOKEN_EXPIRES=3600), nhưng batch ingest tuần tự file scan
    có thể chạy NHIỀU GIỜ — nếu không re-login giữa chừng, mọi request sau 1h sẽ 401."""

    def __init__(self, base_url: str, email: str, password: str):
        self.base_url = base_url
        self.email = email
        self.password = password
        self.token = None

    def login(self, quiet: bool = False):
        resp = requests.post(f"{self.base_url}/api/auth/login",
                             json={"email": self.email, "password": self.password}, timeout=30)
        if resp.status_code != 200:
            print(f"❌ Đăng nhập thất bại ({resp.status_code}): {resp.text[:300]}")
            sys.exit(1)
        self.token = resp.json().get("data", {}).get("access_token")
        if not self.token:
            print(f"❌ Không lấy được access_token: {resp.text[:300]}")
            sys.exit(1)
        if not quiet:
            print(f"✅ Đăng nhập OK ({self.email})")

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def get(self, path: str):
        """GET có tự re-login 1 lần nếu 401 (token hết hạn giữa chừng)."""
        resp = requests.get(f"{self.base_url}{path}", headers=self.headers(), timeout=30)
        if resp.status_code == 401:
            print("   🔑 Token hết hạn — đăng nhập lại...")
            self.login(quiet=True)
            resp = requests.get(f"{self.base_url}{path}", headers=self.headers(), timeout=30)
        return resp


def collect_files(directory: str, recursive: bool) -> list[str]:
    paths = []
    if recursive:
        for root, _dirs, names in os.walk(directory):
            for n in names:
                if os.path.splitext(n)[1].lower() in SUPPORTED_EXT:
                    paths.append(os.path.join(root, n))
    else:
        for n in sorted(os.listdir(directory)):
            full = os.path.join(directory, n)
            if os.path.isfile(full) and os.path.splitext(n)[1].lower() in SUPPORTED_EXT:
                paths.append(full)
    return sorted(paths)


def upload_one(auth: "Auth", path: str, upload_delay: float) -> dict:
    """Upload 1 file. Xử lý 202 (nhận), 409 (trùng), 429 (rate limit → chờ), 401 (token
    hết hạn → đăng nhập lại + thử lại)."""
    filename = os.path.basename(path)
    resp = None
    for attempt in range(5):
        with open(path, "rb") as fh:
            resp = requests.post(
                f"{auth.base_url}/api/documents/upload",
                headers=auth.headers(),
                files={"file": (filename, fh)},
                timeout=120,
            )
        if resp.status_code == 401:
            print(f"   🔑 Token hết hạn khi upload '{filename}' — đăng nhập lại...")
            auth.login(quiet=True)
            continue
        if resp.status_code == 429:
            wait = 60
            print(f"   ⏳ 429 rate-limit ở '{filename}', chờ {wait}s rồi thử lại...")
            time.sleep(wait)
            continue
        break

    body = {}
    try:
        body = resp.json()
    except Exception:
        pass

    time.sleep(upload_delay)  # giãn cách chủ động để tránh dính rate-limit upload (mặc định 10/phút)

    if resp.status_code == 202:
        return {"filename": filename, "http": 202, "doc_id": body.get("doc_id"), "outcome": "accepted"}
    if resp.status_code == 409:
        return {"filename": filename, "http": 409, "doc_id": body.get("existing_doc_id"),
                "outcome": "duplicate", "note": body.get("existing_name")}
    return {"filename": filename, "http": resp.status_code, "doc_id": None,
            "outcome": "upload_error", "note": (body.get("error") if body else resp.text[:200])}


def poll_status(auth: "Auth", doc_id: str) -> dict:
    resp = auth.get(f"/api/documents/{doc_id}")  # tự re-login nếu 401
    if resp.status_code != 200:
        return {"status": "unknown", "_http": resp.status_code}
    return resp.json()


def wait_for_doc(auth: "Auth", doc_id: str, poll_interval: int, timeout: int) -> dict:
    """Chờ 1 document ingest xong (done/failed) hoặc tới timeout. Trả trạng thái cuối.
    Dùng cho chế độ TUẦN TỰ — chỉ 1 file ingest tại một thời điểm, tránh quá tải CPU."""
    deadline = time.time() + timeout
    doc = {}
    while time.time() < deadline:
        doc = poll_status(auth, doc_id)
        if doc.get("status") in ("done", "failed"):
            return doc
        time.sleep(poll_interval)
    return doc  # timeout -> trạng thái cuối cùng


def build_verify_record(up: dict, doc: dict) -> dict:
    """Ghép kết quả upload + trạng thái ingestion cuối → 1 dòng verify + cờ nghi ngờ."""
    rec = {
        "filename": up["filename"],
        "outcome": up["outcome"],
        "doc_id": up.get("doc_id"),
        "status": doc.get("status") if doc else None,
        "page_count": doc.get("page_count") if doc else None,
        "chunk_count": doc.get("chunk_count") if doc else None,
        "flags": [],
    }
    if up.get("note"):
        rec["note"] = up["note"]

    ocr = (doc or {}).get("ocr_stats") or {}
    if ocr:
        needing = ocr.get("pages_needing_ocr", 0) or 0
        failed = ocr.get("pages_ocr_failed", 0) or 0
        rec["ocr"] = {
            "pages_needing_ocr": needing,
            "pages_ocr_failed": failed,
            "ocr_fail_rate": round(failed / needing, 3) if needing else 0.0,
        }
        if needing and (failed / needing) > _SUSPECT_OCR_FAIL_RATE:
            rec["flags"].append("SUSPECT_OCR_FAIL")

    if up["outcome"] == "duplicate":
        rec["flags"].append("DUPLICATE")
    elif up["outcome"] == "upload_error":
        rec["flags"].append("UPLOAD_ERROR")
    elif rec["status"] == "failed":
        rec["flags"].append("INGEST_FAILED")
        rec["error"] = (doc or {}).get("error")
    elif rec["status"] == "done" and (rec["chunk_count"] or 0) < _SUSPECT_MIN_CHUNKS:
        # done nhưng gần như không có chunk → parse/OCR có thể ra rỗng (nghi garbage)
        rec["flags"].append("SUSPECT_LOW_TEXT")
    elif rec["status"] not in ("done", "failed"):
        rec["flags"].append("TIMEOUT_OR_PENDING")

    return rec


def run():
    parser = argparse.ArgumentParser(description="Batch upload + verify tài liệu qua API app đang chạy.")
    parser.add_argument("--dir", required=True, help="Thư mục chứa file cần upload")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--base-url", default="http://localhost:5000")
    parser.add_argument("--recursive", action="store_true", help="Duyệt cả thư mục con")
    parser.add_argument("--upload-delay", type=float, default=6.5,
                        help="Giãn cách giữa 2 upload (giây) — tránh rate-limit 10/phút. Mặc định 6.5s.")
    parser.add_argument("--poll-timeout", type=int, default=2400,
                        help="[--parallel] Tổng thời gian chờ ingestion cả batch (giây)")
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--parallel", action="store_true",
                        help="Upload TẤT CẢ rồi mới poll (hành vi cũ). MẶC ĐỊNH là TUẦN TỰ: "
                             "chờ từng file ingest xong mới upload file kế — tránh dồn nhiều "
                             "thread OCR/embedding cùng lúc làm quá tải CPU và sập tiến trình.")
    parser.add_argument("--file-timeout", type=int, default=900,
                        help="[tuần tự] Thời gian chờ tối đa cho 1 file ingest (giây). Mặc định 900s.")
    parser.add_argument("--report", default=None, help="Đường dẫn lưu report JSON")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    if not os.path.isdir(args.dir):
        print(f"❌ Không tìm thấy thư mục: {args.dir}")
        sys.exit(1)

    files = collect_files(args.dir, args.recursive)
    if not files:
        print(f"❌ Không có file hỗ trợ ({', '.join(sorted(SUPPORTED_EXT))}) trong {args.dir}")
        sys.exit(1)
    print(f"📁 Tìm thấy {len(files)} file hỗ trợ trong {args.dir}\n")

    auth = Auth(base_url, args.email, args.password)
    auth.login()

    uploads = []
    docs_final = {}

    if args.parallel:
        # ── Chế độ SONG SONG (cũ): upload hết rồi mới poll ──
        for i, path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Upload: {os.path.basename(path)}")
            up = upload_one(auth, path, args.upload_delay)
            print(f"     → http={up['http']} outcome={up['outcome']} doc_id={up.get('doc_id')}")
            uploads.append(up)

        pending = {u["doc_id"] for u in uploads if u["outcome"] == "accepted" and u["doc_id"]}
        deadline = time.time() + args.poll_timeout
        print(f"\n⏳ Chờ ingestion cho {len(pending)} file (timeout {args.poll_timeout}s)...")
        while pending and time.time() < deadline:
            time.sleep(args.poll_interval)
            for doc_id in list(pending):
                doc = poll_status(auth, doc_id)
                if doc.get("status") in ("done", "failed"):
                    docs_final[doc_id] = doc
                    pending.discard(doc_id)
                    print(f"   {doc.get('status').upper():<6} {doc.get('filename', doc_id)} "
                          f"(chunks={doc.get('chunk_count')})")
        for doc_id in pending:
            docs_final[doc_id] = poll_status(auth, doc_id)
    else:
        # ── Chế độ TUẦN TỰ (mặc định): upload 1 file -> CHỜ ingest xong -> file kế ──
        # Chỉ 1 file OCR/embedding tại một thời điểm -> không dồn thread làm quá tải CPU
        # (nguyên nhân sập tiến trình khi ingest hàng loạt PDF scan trên máy CPU-only).
        print("🔁 Chế độ TUẦN TỰ — chờ từng file ingest xong mới upload file kế.\n")
        for i, path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Upload: {os.path.basename(path)}")
            # upload_delay=0: tuần tự đã tự giãn (mỗi upload cách nhau cả lần ingest,
            # không thể dính rate-limit 10/phút).
            up = upload_one(auth, path, upload_delay=0)
            print(f"     → http={up['http']} outcome={up['outcome']} doc_id={up.get('doc_id')}")
            uploads.append(up)

            if up["outcome"] == "accepted" and up["doc_id"]:
                doc = wait_for_doc(auth, up["doc_id"], args.poll_interval, args.file_timeout)
                docs_final[up["doc_id"]] = doc
                st = (doc.get("status") or "timeout").upper()
                print(f"     ⏳ ingest: {st} (chunks={doc.get('chunk_count')})")

    # ── Bước 3: Báo cáo verify ──────────────────────────────────
    records = [build_verify_record(up, docs_final.get(up.get("doc_id"), {})) for up in uploads]

    n_done = sum(1 for r in records if r["status"] == "done" and not r["flags"])
    n_done_flagged = sum(1 for r in records if r["status"] == "done" and r["flags"])
    n_dup = sum(1 for r in records if r["outcome"] == "duplicate")
    n_failed = sum(1 for r in records if "INGEST_FAILED" in r["flags"] or "UPLOAD_ERROR" in r["flags"])
    n_pending = sum(1 for r in records if "TIMEOUT_OR_PENDING" in r["flags"])
    total_chunks = sum((r["chunk_count"] or 0) for r in records)

    print("\n" + "=" * 60)
    print("📊 BÁO CÁO VERIFY UPLOAD")
    print("=" * 60)
    print(f"Tổng file gửi     : {len(records)}")
    print(f"  ✅ done (sạch)    : {n_done}")
    print(f"  ⚠️  done (có cờ)   : {n_done_flagged}  ← SOI TAY")
    print(f"  ♻️  trùng (409)    : {n_dup}")
    print(f"  ❌ fail/lỗi upload : {n_failed}")
    print(f"  ⏳ timeout/pending : {n_pending}")
    print(f"Tổng chunk tạo ra : {total_chunks}")

    flagged = [r for r in records if r["flags"] and "DUPLICATE" not in r["flags"]]
    if flagged:
        print("\n-- File CẦN SOI TAY (cờ nghi ngờ) --")
        for r in flagged:
            extra = ""
            if r.get("ocr"):
                extra = f" ocr_fail={r['ocr']['ocr_fail_rate']}"
            print(f"  [{','.join(r['flags'])}] {r['filename']} "
                  f"(status={r['status']}, chunks={r['chunk_count']}){extra}")
    print("=" * 60)

    report_path = args.report or os.path.join(
        os.path.dirname(__file__), "..", "tests", "benchmark", "reports",
        f"upload_verify_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dir": args.dir,
            "summary": {
                "total": len(records), "done_clean": n_done, "done_flagged": n_done_flagged,
                "duplicate": n_dup, "failed": n_failed, "pending": n_pending,
                "total_chunks": total_chunks,
            },
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Report đã lưu: {os.path.abspath(report_path)}")


if __name__ == "__main__":
    run()
