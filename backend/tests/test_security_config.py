"""
Test Secret Management + CORS lockdown (Phase 5 Production Hardening).

SECRET_KEY/JWT_SECRET_KEY/CORS_ALLOWED_ORIGINS đều đọc qua os.getenv() ở CẤP
MODULE (chỉ chạy 1 lần lúc import app.config) — dùng subprocess con với biến
môi trường được kiểm soát chính xác cho từng kịch bản, tránh vấn đề import
cache/reload không đáng tin cậy trong cùng 1 process.

Chạy: python backend/tests/test_security_config.py
"""
import sys, os, subprocess

sys.stdout.reconfigure(encoding="utf-8")
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable


def run_snippet(code: str, env_overrides: dict) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for k in ("SECRET_KEY", "JWT_SECRET_KEY", "CORS_ALLOWED_ORIGINS"):
        env.pop(k, None)
    env.update(env_overrides)
    return subprocess.run(
        [PYTHON, "-c", code],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        # create_app() import kéo theo torch/sentence-transformers (qua
        # blueprint chat.py) — cold import ~45-50s trên máy CPU-only, cần
        # timeout rộng rãi, không phải 30s mặc định.
        timeout=90,
    )


print("=" * 60)
print("TEST 1: ProductionConfig thiếu SECRET_KEY/JWT_SECRET_KEY -> create_app raise RuntimeError")

code1 = """
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('.')
from app import create_app
try:
    create_app('production')
    print('NO_ERROR')
except RuntimeError as e:
    print('RUNTIME_ERROR:' + str(e))
"""
r = run_snippet(code1, {})
out = r.stdout.strip()
assert "RUNTIME_ERROR" in out, f"❌ Kỳ vọng RuntimeError, nhận: stdout={out!r} stderr={r.stderr!r}"
assert "SECRET_KEY" in out and "JWT_SECRET_KEY" in out, f"❌ Thông báo lỗi phải liệt kê đủ 2 biến thiếu: {out}"
print(f"  {out}")
print("  ✅ PASS\n")

print("=" * 60)
print("TEST 2: ProductionConfig ĐỦ SECRET_KEY/JWT_SECRET_KEY -> create_app thành công")

code2 = """
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('.')
from app import create_app
app = create_app('production')
print('OK:' + app.config['SECRET_KEY'][:8])
"""
# ALLOW_SINGLE_WORKER_PERSISTENT=true để cô lập kiểm tra SECRET khỏi guard Chroma
# persistent mới thêm (2026-07-22, xem __init__.py) — test này chỉ xác nhận đủ
# secret thì create_app('production') chạy, không phải test hành vi Chroma.
r2 = run_snippet(code2, {"SECRET_KEY": "a" * 64, "JWT_SECRET_KEY": "b" * 64,
                         "ALLOW_SINGLE_WORKER_PERSISTENT": "true"})
assert r2.returncode == 0, f"❌ create_app('production') phải thành công khi đủ secret: {r2.stdout} {r2.stderr}"
assert "OK:" in r2.stdout, f"❌ {r2.stdout} {r2.stderr}"
print(f"  {r2.stdout.strip()}")
print("  ✅ PASS\n")

print("=" * 60)
print("TEST 3: DevelopmentConfig/TestingConfig vẫn chạy được KHÔNG cần set secret (fallback riêng)")

code3 = """
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('.')
from app import create_app
app_dev = create_app('development')
app_test = create_app('testing')
print('dev_has_secret:', bool(app_dev.config['SECRET_KEY']))
print('testing_has_secret:', bool(app_test.config['SECRET_KEY']))
"""
r3 = run_snippet(code3, {})
assert r3.returncode == 0, f"❌ {r3.stdout} {r3.stderr}"
assert "dev_has_secret: True" in r3.stdout, r3.stdout
assert "testing_has_secret: True" in r3.stdout, r3.stdout
print(f"  {r3.stdout.strip()}")
print("  ✅ PASS\n")

print("=" * 60)
print("TEST 4: CORS_ALLOWED_ORIGINS — mặc định chỉ localhost, tuỳ chỉnh parse đúng nhiều origin")

code4 = """
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('.')
from app.config import BaseConfig
print(BaseConfig.CORS_ALLOWED_ORIGINS)
"""
r4a = run_snippet(code4, {})
assert r4a.stdout.strip() == "['http://localhost:5173']", f"❌ Mặc định sai: {r4a.stdout}"
print(f"  Mặc định: {r4a.stdout.strip()} ✅")

r4b = run_snippet(code4, {"CORS_ALLOWED_ORIGINS": "http://a.com, http://b.com ,http://c.com"})
assert r4b.stdout.strip() == "['http://a.com', 'http://b.com', 'http://c.com']", f"❌ Parse nhiều origin sai: {r4b.stdout}"
print(f"  Nhiều origin (có khoảng trắng thừa): {r4b.stdout.strip()} ✅")
print("  ✅ PASS\n")

print("=" * 60)
print("🎉 TEST SECURITY CONFIG HOÀN THÀNH — TẤT CẢ PASS!")
