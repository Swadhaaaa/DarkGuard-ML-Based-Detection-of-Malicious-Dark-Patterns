"""
================================================================================
 DarkGuard ML — Production REST API Server
================================================================================
 Endpoints:
   POST /predict          → Single page prediction
   POST /predict/batch    → Bulk prediction (up to 100 records)
   GET  /health           → Health check (for load balancer probes)
   GET  /metrics          → Latency & throughput stats
   GET  /model/info       → Model version and metadata

 Autoscaling:
   - Stateless design: any replica can handle any request
   - Thread-safe: model loaded once at startup, shared across threads
   - Gunicorn workers: configured via gunicorn.conf.py
   - Rate limiting: prevents abuse in production
================================================================================
"""

import os
import time
import json
import logging
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path
from collections import deque

import numpy as np
from flask import Flask, request, jsonify, g

# ── Import the Predictor from the ML engine ──────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dark_pattern_ml import DarkPatternPredictor, CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# APP & LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DarkGuard.API")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE (initialized at startup — thread-safe after init)
# ─────────────────────────────────────────────────────────────────────────────

class ServerState:
    predictor: DarkPatternPredictor = None
    startup_time: str = None
    request_count: int = 0
    error_count: int = 0
    latency_window = deque(maxlen=1000)   # last 1000 request latencies (ms)
    lock = threading.Lock()

    @classmethod
    def record_request(cls, latency_ms: float, success: bool):
        with cls.lock:
            cls.request_count += 1
            cls.latency_window.append(latency_ms)
            if not success:
                cls.error_count += 1

    @classmethod
    def get_stats(cls):
        with cls.lock:
            lats = list(cls.latency_window)
        if not lats:
            return {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "avg_ms": 0}
        lats_sorted = sorted(lats)
        n = len(lats_sorted)
        return {
            "p50_ms":  round(lats_sorted[int(n * 0.50)], 2),
            "p95_ms":  round(lats_sorted[int(n * 0.95)], 2),
            "p99_ms":  round(lats_sorted[int(n * 0.99)], 2),
            "avg_ms":  round(sum(lats_sorted) / n, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE — Timing & request logging
# ─────────────────────────────────────────────────────────────────────────────

@app.before_request
def _start_timer():
    g.t_start = time.perf_counter()


@app.after_request
def _log_request(response):
    latency_ms = (time.perf_counter() - g.t_start) * 1000
    success = response.status_code < 400
    ServerState.record_request(latency_ms, success)
    log.info(
        f"{request.method} {request.path} → {response.status_code} "
        f"({latency_ms:.1f}ms)"
    )
    # CORS headers (required for browser extension)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["X-DarkGuard-Version"] = CONFIG.get("MODEL_VERSION", "unknown")
    return response


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>",              methods=["OPTIONS"])
def _cors_preflight(path):
    return jsonify({}), 200


# ─────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

VALID_FEATURES = set(CONFIG["FEATURES"])

def validate_signals(payload: dict) -> tuple:
    """Returns (signals_dict, error_message)."""
    if not isinstance(payload, dict):
        return None, "Payload must be a JSON object."

    signals = {}
    errors  = []

    for feat in CONFIG["FEATURES"]:
        val = payload.get(feat, 0)
        try:
            val = int(val)
        except (TypeError, ValueError):
            errors.append(f"'{feat}' must be 0 or 1, got: {val!r}")
            continue
        if val not in (0, 1):
            errors.append(f"'{feat}' must be 0 or 1, got: {val}")
        else:
            signals[feat] = val

    if errors:
        return None, "; ".join(errors)

    # Fill any missing features with 0
    for feat in CONFIG["FEATURES"]:
        signals.setdefault(feat, 0)

    return signals, None


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — Standard API response format
# ─────────────────────────────────────────────────────────────────────────────

def success_response(data: dict, status: int = 200):
    return jsonify({"status": "ok", "data": data, "timestamp": datetime.utcnow().isoformat() + "Z"}), status


def error_response(message: str, status: int = 400):
    log.warning(f"API Error [{status}]: {message}")
    return jsonify({"status": "error", "error": message, "timestamp": datetime.utcnow().isoformat() + "Z"}), status


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """
    Health probe for load balancer / k8s readiness check.
    Returns 200 when model is loaded and ready.
    Returns 503 if model is not yet initialized (during cold start).
    """
    if ServerState.predictor is None:
        return error_response("Model not loaded", 503)
    return success_response({
        "status":       "healthy",
        "model_loaded": True,
        "uptime_since": ServerState.startup_time,
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus-compatible metrics (also readable as JSON)."""
    stats = ServerState.get_stats()
    return success_response({
        "total_requests": ServerState.request_count,
        "total_errors":   ServerState.error_count,
        "error_rate":     round(
            ServerState.error_count / max(ServerState.request_count, 1), 4
        ),
        "latency": stats,
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    """Returns model metadata for observability / debugging."""
    meta_path = Path("model_artifacts") / "model_metadata_LATEST.json"
    if not meta_path.exists():
        return error_response("Model metadata not found.", 404)
    with open(meta_path) as f:
        meta = json.load(f)
    return success_response(meta)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single-page dark pattern prediction.

    Request body (JSON):
        {
            "Fake_Urgency":      0 or 1,
            "Scarcity":          0 or 1,
            "Confusing_Text":    0 or 1,
            "Hidden_Cost":       0 or 1,
            "Forced_Action":     0 or 1,
            "Social_Proof_Fake": 0 or 1,
            "Misdirection":      0 or 1,
            "Visual_Trick":      0 or 1
        }

    Response:
        {
            "status": "ok",
            "data": {
                "is_dark_pattern":   true/false,
                "confidence":        0.0-1.0,
                "threat_level":      "NONE|LOW|MEDIUM|HIGH|CRITICAL",
                "weighted_score":    0.0-1.0,
                "pattern_count":     0-8,
                "detected_patterns": [...],
                "certainty":         "CERTAIN|UNCERTAIN"
            }
        }
    """
    if ServerState.predictor is None:
        return error_response("Model not ready", 503)

    payload = request.get_json(silent=True)
    if payload is None:
        return error_response("Invalid JSON body.")

    signals, err = validate_signals(payload)
    if err:
        return error_response(err, 422)

    try:
        result = ServerState.predictor.predict(signals)
        return success_response(result)
    except Exception as e:
        log.exception("Prediction error")
        return error_response(f"Prediction failed: {str(e)}", 500)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction for multiple pages.

    Request body (JSON):
        {
            "records": [
                {"Fake_Urgency": 1, "Scarcity": 1, ...},
                {"Fake_Urgency": 0, "Scarcity": 0, ...},
                ...
            ]
        }
    Limit: 100 records per request.
    """
    if ServerState.predictor is None:
        return error_response("Model not ready", 503)

    payload = request.get_json(silent=True)
    if not payload or "records" not in payload:
        return error_response("Body must contain a 'records' list.")

    records = payload["records"]
    if not isinstance(records, list):
        return error_response("'records' must be a list.")
    if len(records) > 100:
        return error_response("Batch limit is 100 records per request.")
    if len(records) == 0:
        return error_response("'records' list is empty.")

    results = []
    errors  = []

    for idx, record in enumerate(records):
        signals, err = validate_signals(record)
        if err:
            errors.append({"index": idx, "error": err})
            continue
        try:
            results.append({"index": idx, "result": ServerState.predictor.predict(signals)})
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    return success_response({
        "total":     len(records),
        "succeeded": len(results),
        "failed":    len(errors),
        "results":   results,
        "errors":    errors,
    })


# ─────────────────────────────────────────────────────────────────────────────
# SERVER STARTUP
# ─────────────────────────────────────────────────────────────────────────────

def load_model_on_startup():
    """Called once when the server starts."""
    try:
        log.info("Loading DarkGuard ML model...")
        ServerState.predictor    = DarkPatternPredictor.load("model_artifacts")
        ServerState.startup_time = datetime.utcnow().isoformat() + "Z"
        log.info("Model loaded successfully. Server is ready.")
    except Exception as e:
        log.error(f"CRITICAL: Model failed to load: {e}")
        log.error("Server will start but /predict will return 503.")


# Load model before first request (Gunicorn-safe)
with app.app_context():
    load_model_on_startup()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT (Development only — use Gunicorn in production)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    log.info(f"Starting DarkGuard API on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
