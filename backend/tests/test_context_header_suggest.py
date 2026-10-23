"""
Test gợi ý header ngữ cảnh (app/core/context_header_suggest.py) + chuyển trạng thái
duyệt/gợi ý (app/services/context_header_service.py::plan_header_update) + bước gợi ý lúc
ingest (D1-B). Văn bản mẫu mô phỏng ĐÚNG các ca khó gặp trên 89 tài liệu thật (2026-10-23):
âm tiết bị tách do chữ ký số, công ty con trong thuyết minh, số hiệu văn bản được trích dẫn,
bản ký số mất trang tiêu đề. Không cần Ollama/MongoDB/ChromaDB (SLM được mock).

Chạy: python backend/tests/test_context_header_suggest.py
"""
import sys, os, json, tempfile
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app.core.context_header_suggest as sg
from app.core import context_header as ctx
from app.core.context_header_suggest import normalize_header, repair_split_syllables, suggest_header, ordered_texts
from app.services.context_header_service import plan_header_update, header_state, effective_header
from app.ingestion.pipeline import IngestionPipeline


def _refuse_slm(*a, **kw):
    raise AssertionError("Rule đủ trường thì KHÔNG được gọi SLM")


sg.requests.post = _refuse_slm

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: normalize_header / repair_split_syllables")

assert normalize_header("  Báo cáo   năm 2016 ") == "[Báo cáo năm 2016]"
assert normalize_header("[đã có ngoặc]") == "[đã có ngoặc]"
assert normalize_header("dòng 1\ndòng 2") == "[dòng 1 dòng 2]"
assert normalize_header("") is None and normalize_header("[ ]") is None and normalize_header(None) is None
try:
    normalize_header("x" * 300)
    raise AssertionError("header quá dài phải bị từ chối")
except ValueError:
    pass
assert repair_split_syllables("T Ổ NG CÔNG TY C Ổ PH Ầ N XU Ấ T NH Ậ P KH Ẩ U VI Ệ T NAM") == \
    "TỔNG CÔNG TY CỔ PHẦN XUẤT NHẬP KHẨU VIỆT NAM"
assert repair_split_syllables("Xây d ự ng Vi ệ t Nam") == "Xây dựng Việt Nam"
assert repair_split_syllables("L= Đố ng Đ a, ổ cứng") == "L= Đố ng Đ a, ổ cứng", "Đ (phụ âm) và từ 1 âm tiết không được ghép"
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: Excel có dòng thông tin (Simplize)")

excel = ["[Sheet: Sheet 1] [Dòng 2-11] | Tên công ty | Công ty Cổ phần Đầu tư Thế Giới Di Động | | |\n"
         "| Tên báo cáo | Lịch sử giá | | |\n| NGÀY | GIÁ MỞ CỬA | KHỐI LƯỢNG |"]
s = suggest_header("Simplize_MWG_PriceHistory_20250315.xlsx", excel)
assert (s.source, s.kind, s.needs_review) == ("rule", "excel_table", False), s
assert s.header == "[Lịch sử giá của Công ty Cổ phần Đầu tư Thế Giới Di Động (mã CK: MWG) — nguồn Simplize]", s.header
print(f"  {s.header}\n  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: Báo cáo tài chính — đơn vị ký số thắng công ty con trong thuyết minh")

subsidiary_noise = " ".join(["Báo cáo tài chính của Công ty TNHH Một thành viên Giáo dục Lý Thái Tổ được hợp nhất."] * 6)
bctc = [
    "Digitally signed by T Ổ NG CÔNG TY C Ổ PH Ầ N XU Ấ T NH Ậ P KH Ẩ U VÀ XÂY D Ự NG VI Ệ T NAM DN: C=VN",
    "Báo cáo tài chính hợp nhất cho năm tài chính kết thúc ngày 31 tháng 12 năm 2023 của Vinaconex.",
    subsidiary_noise,
    "Báo cáo tài chính hợp nhất. Vinaconex. Vinaconex. Số liệu so sánh tại ngày 31 tháng 12 năm 2022.",
]
s = suggest_header("2023-0100105616-20240329-523-BCTC-vinaconex.pdf", bctc)
assert (s.source, s.kind, s.needs_review) == ("rule", "financial_report", False), s
assert s.fields["year"] == "2023" and s.fields["report_kind"] == "hợp nhất"
assert "Giáo dục Lý Thái Tổ" not in s.header and "Xuất Nhập Khẩu" in s.header and "(Vinaconex)" in s.header, s.header

s = suggest_header("2016-0100105616-228721-BCTCR-vinaconex.pdf", bctc)
assert s.fields["report_kind"] == "riêng", "BCTCR trong tên file = báo cáo riêng"

vinahud = ["Ký bởi: CÔNG TY CỔ PHẦN ĐẦU TƯ PHÁT TRIỂN NHÀ VÀ ĐÔ THỊ VINAHUD Ký ngày: 15/3/2022",
           "Báo cáo tài chính cho năm tài chính kết thúc ngày 31 tháng 12 năm 2021"]
s = suggest_header("2021-0102294285-20220314-1730-BCTCvinahud.pdf", vinahud)
assert s.fields["report_kind"] is None and s.needs_review, "không ghi riêng/hợp nhất → cần người duyệt"
assert "Vinahud" in s.header and "Ký ngày" not in s.header, s.header
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: Văn bản pháp luật — số hiệu CỦA CHÍNH văn bản, không phải văn bản được trích dẫn")

law = ["QUỐC HỘI Luật số: 24/2018/QH14 CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM LUẬT AN NINH MẠNG "
       "Căn cứ Hiến pháp; Căn cứ Nghị định số 171/2016/NĐ-CP ..."]
s = suggest_header("24-2018-qh14.pdf", law)
assert s.header == "[Luật An ninh mạng số 24/2018/QH14]", s.header

decree = ["CHÍNH PHỦ Số: 145/2020/NĐ-CP NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT MỘT SỐ ĐIỀU CỦA BỘ LUẬT LAO ĐỘNG "
          "VỀ ĐIỀU KIỆN LAO ĐỘNG Căn cứ Bộ luật Lao động ngày 20 tháng 11 năm 2019;"]
s = suggest_header("1452020nd-cp.docx", decree)
assert s.header.startswith("[Nghị định số 145/2020/NĐ-CP quy định chi tiết một số điều của bộ luật lao động"), \
    f"phải lấy trích yếu sau NGHỊ ĐỊNH, không phải sau chữ LUẬT bên trong: {s.header}"

consolidated = ["BỘ LUẬT LAO ĐỘNG Bộ luật Lao động số 45/2019/QH14 ngày 20 tháng 11 năm 2019 của Quốc hội, "
                "có hiệu lực kể từ ngày 01 tháng 01 năm 2021, được sửa đổi, bổ sung bởi: Luật số 71/2025/QH15"]
s = suggest_header("luat-so-125.pdf", consolidated)
assert s.header == "[Bộ luật Lao động số 45/2019/QH14 (bản hợp nhất)]", s.header

charter = ["ĐIỀU LỆ CÔNG TY CỔ PHẦN ... Căn cứ vào Luật Doanh nghiệp số 59/2020/QH14 được Quốc hội thông qua"]
s = suggest_header("mau-dieu-le-cong-ty-co-phan-mau-1.docx", charter, slm_enabled=False)
assert s.header is None and s.needs_review, f"điều lệ trích dẫn Luật DN không được thành 'Luật số 59/2020': {s.header}"

signed_mid = ["Ký bởi: Cổng Thông tin điện tử Chính phủ a) Lao động, việc làm;", "Nghị định này ...",
              "Nghị định số 70/2025/NĐ-CP có hiệu lực. Căn cứ Nghị định số 123/2020/NĐ-CP."]
s = suggest_header("70-nd-cp.signed.pdf", signed_mid)
assert s.fields["number"] == "70/2025/NĐ-CP" and s.needs_review, "bản ký số mất tiêu đề: số hiệu khớp tên file, cần duyệt"
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 5: SLM fallback — code tự ráp header, lỗi SLM → None (không raise)")


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


calls = []
sg.requests.post = lambda url, json=None, timeout=None: calls.append(json) or FakeResponse(
    '{"loai": "Tài liệu ôn tập", "ten": "Kiến trúc máy tính - CO2008", "don_vi": "Trường Đại học Bách Khoa", "nam": "2022"}')
s = suggest_header("ktmt_OnTap.pdf", ["TRƯỜNG ĐẠI HỌC BÁCH KHOA ÔN TẬP Kiến trúc máy tính - CO2008 August, 2022"],
                   slm_enabled=True)
assert (s.source, s.needs_review) == ("slm", True), s
assert s.header == "[Tài liệu ôn tập: Kiến trúc máy tính - CO2008 — Trường Đại học Bách Khoa, 2022]", s.header
assert calls and calls[0]["format"] == "json" and calls[0]["options"]["temperature"] == 0.0

sg.requests.post = lambda *a, **kw: FakeResponse('{"loai": "", "ten": "", "nam": "khoảng 2020"}')
s = suggest_header("x.pdf", ["nội dung không rõ"], slm_enabled=True)
assert s.header is None and s.source is None, "SLM không trích được tên → không có header"

# Ca thật dry-run 2026-10-23: SLM trả lại tên file / "không rõ" / chép loại sai lên tiêu đề đã có loại
sg.requests.post = lambda *a, **kw: FakeResponse('{"loai": "Quy trình", "ten": "45.signed.pdf", "don_vi": "không rõ"}')
assert suggest_header("45.signed.pdf", ["a) Lao động ..."], slm_enabled=True).header is None
sg.requests.post = lambda *a, **kw: FakeResponse(
    '{"loai": "Quy trình", "ten": "Luật Thương mại 2005", "don_vi": "Quốc hội", "nam": "2005"}')
s = suggest_header("luat-thuong-mai-2005.pdf", ["LUẬT THƯƠNG MẠI ..."], slm_enabled=True)
assert s.header == "[Luật Thương mại 2005 — Quốc hội, 2005]", s.header


def _boom(*a, **kw):
    raise ConnectionError("Ollama tắt")


sg.requests.post = _boom
s = suggest_header("x.pdf", ["nội dung"], slm_enabled=True)
assert s.header is None and s.needs_review, "Ollama lỗi → fail-open"
sg.requests.post = _refuse_slm
s = suggest_header("x.pdf", ["nội dung"], slm_enabled=False)
assert s.header is None
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 6: plan_header_update — version chỉ tăng khi header HIỆU LỰC đổi")

H1, H2 = "[Báo cáo tài chính riêng năm 2016 của A]", "[Báo cáo tài chính hợp nhất năm 2016 của A]"
doc = {"_id": "d1"}
u = plan_header_update(doc, suggested=H1, suggested_source="rule", needs_review=False)
assert (u["context_header_version"], u["context_header_source"]) == (1, "rule"), u
doc.update(u)
assert effective_header(doc) == H1 and header_state(doc)["status"] == "suggested"

u = plan_header_update(doc, approved=H1)  # duyệt nguyên văn → embedding không đổi
assert u["context_header_version"] == 1 and u["context_header_source"] == "rule", u
doc.update(u)
assert header_state(doc)["status"] == "approved"

u = plan_header_update(doc, approved=H2)  # người dùng sửa → đổi chuỗi → cần embed lại
assert (u["context_header_version"], u["context_header_source"]) == (2, "user"), u
doc.update(u)

u = plan_header_update(doc, suggested="[gợi ý mới]", suggested_source="slm")  # gợi ý lại KHÔNG đè bản duyệt
assert u["context_header"] == H2 and u["context_header_version"] == 2, u
doc.update(u)

u = plan_header_update(doc, approved=None)  # bỏ bản duyệt → quay về bản gợi ý
assert u["context_header_version"] == 3 and u["context_header_source"] == "slm", u
doc.update(u)
state = header_state({**doc, "embedded_header_version": 1})
assert state["effective"] == "[gợi ý mới]" and state["needs_reembed"] is True
print("  ✅ PASS\n")

# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 7: Ingest dùng ngay header gợi ý (D1-B) + ghi nhận embedding đã phản ánh header chưa")


class RecordingEmbedder:
    def __init__(self):
        self.texts = []

    def embed_batch(self, texts):
        self.texts.extend(texts)
        return [[0.0, 0.0, 0.0] for _ in texts]


class FakeVectorOps:
    def __init__(self):
        self.stored = []

    def add_chunks(self, chunks):
        self.stored.extend(chunks)


class FakeDocService:
    def update_status(self, doc_id, status, **kwargs):
        self.status = status


class FakeHeaderService:
    def __init__(self, header, fail=False):
        self.header, self.fail, self.recorded = header, fail, None

    def suggest_from_chunks(self, filename, chunks):
        if self.fail:
            raise RuntimeError("gợi ý lỗi")
        assert chunks and "text" in chunks[0]
        return sg.HeaderSuggestion(self.header, "rule", "legal")

    def record_ingest(self, doc_id, suggestion, embedded_header):
        self.recorded = (suggestion.header if suggestion else None, embedded_header)


def run_ingest(enabled, header_service):
    ctx.set_enabled(enabled)
    fd, path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("# Thuyết minh\n\nDoanh thu thuần năm nay đạt 1.234 tỷ đồng, tăng so với năm trước.\n")
    embedder, ops, docs = RecordingEmbedder(), FakeVectorOps(), FakeDocService()
    ok = IngestionPipeline(vector_ops=ops, doc_service=docs, embedder=embedder,
                           header_service=header_service).ingest(path, "doc1", "nd.md")
    os.remove(path)
    ctx.set_enabled(False)
    return ok, docs.status, embedder.texts, ops.stored


HDR = "[Nghị định số 1/2026/NĐ-CP]"
hs = FakeHeaderService(HDR)
ok, status, texts, stored = run_ingest(True, hs)
assert ok and status == "done"
assert all(t.startswith(HDR + "\n") for t in texts) and stored[0]["metadata"]["context_header"] == HDR
assert hs.recorded == (HDR, HDR), hs.recorded

hs = FakeHeaderService(HDR)
ok, status, texts, stored = run_ingest(False, hs)
assert ok and not texts[0].startswith("[") and "context_header" not in stored[0]["metadata"]
assert hs.recorded == (HDR, None), "killswitch tắt: vẫn lưu gợi ý, nhưng ghi nhận embedding CHƯA có header"

hs = FakeHeaderService(HDR, fail=True)
ok, status, texts, stored = run_ingest(True, hs)
assert ok and status == "done", "gợi ý lỗi không được làm hỏng ingest"
assert not texts[0].startswith("[") and hs.recorded == (None, None)
print("  ✅ PASS\n")

print("🎉 TẤT CẢ TEST GỢI Ý HEADER PASS!")
