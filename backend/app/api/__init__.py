# ==============================================================
# app/api/__init__.py — API Package, export tất cả blueprints
# ==============================================================

from .auth import auth_bp
from .documents import documents_bp
from .chat import chat_bp
from .collections import collections_bp
from .health import health_bp

__all__ = ["auth_bp", "documents_bp", "chat_bp", "collections_bp", "health_bp"]
