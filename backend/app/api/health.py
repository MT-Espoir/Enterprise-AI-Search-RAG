from flask import Blueprint, jsonify
from ..extensions import mongo

health_bp = Blueprint("health", __name__)

@health_bp.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    status = {
        "status": "ok",
        "services": {
            "mongodb": "unknown"
        }
    }
    
    # Try pinging MongoDB
    try:
        if mongo.cx is not None:
            mongo.db.command("ping")
            status["services"]["mongodb"] = "connected"
        else:
            status["services"]["mongodb"] = "disconnected"
    except Exception as e:
        status["services"]["mongodb"] = "error"
        status["status"] = "degraded"
        status["error"] = str(e)
        
    return jsonify(status), 200 if status["status"] == "ok" else 503
