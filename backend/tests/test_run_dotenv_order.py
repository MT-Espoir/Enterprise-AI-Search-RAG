"""
Test run.py nạp .env TRƯỚC khi import app (app/config.py đọc os.getenv ngay lúc import).

Lỗi cũ: run.py `from app import create_app` rồi mới `load_dotenv()` → chạy không debug (không có reloader để tiến
trình con kế thừa env) thì MỌI killswitch trong .env về mặc định — phát hiện ở live E2E M5 (2026-10-26).

Cách test (subprocess, không đụng .env / DB thật): chép run.py vào thư mục tạm cùng 1 file .env có biến mốc và 1 gói
`app` giả ghi lại giá trị biến mốc ĐÚNG LÚC bị import; chạy run.py bằng runpy (không phải __main__ → không app.run()).

Chạy: python backend/tests/test_run_dotenv_order.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

RUN_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "run.py"))
SENTINEL = "RUN_PY_DOTENV_ORDER_SENTINEL"

FAKE_APP = f'''import os
SEEN_AT_IMPORT = os.environ.get("{SENTINEL}")


class _App:
    def run(self, **kwargs):
        raise AssertionError("không được gọi app.run() khi import")


def create_app(env):
    return _App()
'''

CHECK = f'''import runpy, sys
runpy.run_path("run.py", run_name="run_under_test")
seen = sys.modules["app"].SEEN_AT_IMPORT
print("SEEN=" + str(seen))
sys.exit(0 if seen == "loaded" else 1)
'''

print("=" * 60)
print("TEST: run.py nạp .env trước khi import app")
tmp = tempfile.mkdtemp()
try:
    shutil.copy(RUN_PY, os.path.join(tmp, "run.py"))
    with open(os.path.join(tmp, ".env"), "w", encoding="utf-8") as f:
        f.write(f"{SENTINEL}=loaded\n")
    os.makedirs(os.path.join(tmp, "app"))
    with open(os.path.join(tmp, "app", "__init__.py"), "w", encoding="utf-8") as f:
        f.write(FAKE_APP)
    env = {k: v for k, v in os.environ.items() if k != SENTINEL}
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, "-c", CHECK], cwd=tmp, env=env, capture_output=True, text=True,
                            encoding="utf-8")
    assert result.returncode == 0, f"app import thấy biến .env = {result.stdout.strip()!r} {result.stderr[-500:]}"
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("  ✅ PASS — app/config thấy biến .env ngay lúc import\n")
print("🎉 TEST THỨ TỰ NẠP .env PASS!")
