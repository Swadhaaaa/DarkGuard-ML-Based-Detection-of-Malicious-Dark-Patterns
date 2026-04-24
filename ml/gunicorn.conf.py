# ============================================================
#  Gunicorn Production Configuration — DarkGuard ML API
#  Autoscaling + Performance + Reliability settings
# ============================================================
#
#  Usage:
#    gunicorn -c gunicorn.conf.py api_server:app
#
# ============================================================

import multiprocessing
import os

# ── Server Socket ─────────────────────────────────────────
bind            = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog         = 2048          # Max pending connections in queue

# ── Worker Processes (Autoscaling Formula) ────────────────
# Industry standard: (2 × CPU cores) + 1
# On a 4-core machine → 9 workers
# Each handles one request at a time; OS schedules between them
workers         = int(os.environ.get(
    "WEB_CONCURRENCY",
    multiprocessing.cpu_count() * 2 + 1
))

# ── Worker Type ── ────────────────────────────────────────
# "gthread" = multi-threaded workers; ideal for I/O-bound ML inference
# Each worker spawns `threads` threads → total concurrency = workers × threads
worker_class    = "gthread"
threads         = int(os.environ.get("WORKER_THREADS", "4"))
worker_connections = 1000       # Max simultaneous clients per worker

# ── Timeouts ─────────────────────────────────────────────
# ML inference is fast (<50ms); give generous timeout for cold starts
timeout         = 120           # Kill worker if silent for 120s
keepalive       = 5             # Keep-alive connections for 5s
graceful_timeout = 30           # Time to finish in-flight requests on shutdown

# ── Auto-restart (Memory Leak Protection) ─────────────────
# Recycle workers after handling N requests to prevent memory bloat
max_requests               = 1000
max_requests_jitter        = 100    # Randomise restart to avoid thundering herd

# ── Process Naming ────────────────────────────────────────
proc_name       = "darkguard-ml-api"

# ── Logging ──────────────────────────────────────────────
loglevel        = os.environ.get("LOG_LEVEL", "info")
accesslog       = "-"           # stdout
errorlog        = "-"           # stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" %(D)sms'

# ── Security ─────────────────────────────────────────────
limit_request_line   = 4096     # Max HTTP request line length
limit_request_fields = 100      # Max number of HTTP headers
limit_request_field_size = 8190

# ── Hooks — Lifecycle callbacks ───────────────────────────

def on_starting(server):
    server.log.info("=" * 60)
    server.log.info("  DarkGuard ML API — Starting Up")
    server.log.info(f"  Workers  : {workers}")
    server.log.info(f"  Threads  : {threads}")
    server.log.info(f"  Bind     : {bind}")
    server.log.info("=" * 60)


def post_fork(server, worker):
    """Called after each worker process is forked."""
    server.log.info(f"Worker spawned (pid={worker.pid})")


def worker_int(worker):
    """Called when a worker receives SIGINT (graceful stop)."""
    worker.log.info(f"Worker interrupted (pid={worker.pid}) — finishing in-flight requests")


def worker_exit(server, worker):
    """Called when a worker exits."""
    server.log.info(f"Worker exited (pid={worker.pid})")


def on_exit(server):
    server.log.info("DarkGuard ML API shutting down cleanly.")
