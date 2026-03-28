"""
check_langfuse.py — Langfuse v2 connectivity diagnostic script.

Loads credentials from .env (same as app.py) and runs a series of checks:
  1. Env vars present
  2. Host reachable (HTTP)
  3. Auth valid (auth_check — returns bool in v2)
  4. Write trace (lf.trace + flush)
  5. Read trace back via fetch_trace

Usage:
    python check_langfuse.py
"""

import sys
import os
import uuid
import time

# ── Load .env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(override=True)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

def _ok(label: str, detail: str = "") -> None:
    print(f"  {PASS}  {label}" + (f"  →  {detail}" if detail else ""))

def _fail(label: str, detail: str = "") -> None:
    print(f"  {FAIL}  {label}" + (f"\n       {detail}" if detail else ""))

def _warn(label: str, detail: str = "") -> None:
    print(f"  {WARN} {label}" + (f"  →  {detail}" if detail else ""))

# ── Step 1: env vars ──────────────────────────────────────────────────────────
print("\n[1/5] Checking environment variables…")

public_key  = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
secret_key  = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
host        = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
project_id  = os.environ.get("LANGFUSE_PROJECT_ID", "").strip()

missing = []
for name, val in [("LANGFUSE_PUBLIC_KEY", public_key), ("LANGFUSE_SECRET_KEY", secret_key)]:
    if val:
        _ok(name, val[:12] + "…")
    else:
        _fail(name, "not set")
        missing.append(name)

_ok("LANGFUSE_HOST", host)
if project_id:
    _ok("LANGFUSE_PROJECT_ID", project_id)
else:
    _warn("LANGFUSE_PROJECT_ID", "not set (optional — deep-link URLs will not work)")

if missing:
    print(f"\n{FAIL}  Aborting: required variable(s) missing: {', '.join(missing)}")
    sys.exit(1)

# ── Step 2: HTTP reachability ─────────────────────────────────────────────────
print("\n[2/5] Checking host reachability…")
import httpx

health_url = host.rstrip("/") + "/api/public/health"
try:
    r = httpx.get(health_url, timeout=60, follow_redirects=True)
    if r.status_code == 200:
        _ok(f"GET {health_url}", f"HTTP {r.status_code}")
    else:
        _warn(f"GET {health_url}", f"HTTP {r.status_code} — host is up but health endpoint returned non-200")
except Exception as exc:
    _fail(f"GET {health_url}", str(exc))
    print(f"\n{FAIL}  Aborting: cannot reach Langfuse host.")
    sys.exit(1)

# ── Step 3: auth_check ────────────────────────────────────────────────────────
print("\n[3/5] Verifying credentials (auth_check)…")

os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
os.environ["LANGFUSE_SECRET_KEY"] = secret_key
os.environ["LANGFUSE_HOST"]       = host

from langfuse import Langfuse

lf = Langfuse(
    public_key=public_key,
    secret_key=secret_key,
    host=host,
)

# In Langfuse v2, auth_check() returns a boolean directly (no Pydantic parsing).
auth_ok = False
try:
    if lf.auth_check():
        _ok("auth_check passed")
        auth_ok = True
    else:
        # auth_check() returned False — confirm with raw HTTP probe
        probe = httpx.get(
            host.rstrip("/") + "/api/public/projects",
            auth=(public_key, secret_key),
            timeout=60,
            follow_redirects=True,
        )
        if probe.status_code == 200:
            _warn("auth_check() returned False but HTTP probe succeeded — check SDK/server version match")
            auth_ok = True
        else:
            _fail("auth_check failed", f"HTTP probe returned {probe.status_code}")
except Exception as exc:
    _fail("auth_check raised an exception", str(exc))

if not auth_ok:
    print(f"\n{FAIL}  Aborting: credentials invalid.")
    sys.exit(1)

# ── Step 4: write a test trace ────────────────────────────────────────────────
print("\n[4/5] Writing a test trace…")

# In Langfuse v2, lf.trace() creates (or upserts) a trace and returns a
# StatefulTraceClient. Calling .update() or .generation() on it adds children.
trace_id = str(uuid.uuid4())
try:
    trace = lf.trace(
        id=trace_id,
        name="check_langfuse_connectivity",
        input={"test": True},
        output={"result": "connectivity check"},
        metadata={"source": "check_langfuse.py"},
        tags=["connectivity-check"],
    )
    lf.flush()
    _ok("Trace created + flushed", f"trace_id={trace_id}")
except Exception as exc:
    _fail("Failed to create trace", str(exc))
    sys.exit(1)

# ── Step 5: read trace back ───────────────────────────────────────────────────
print("\n[5/5] Reading trace back via SDK…")

# Langfuse ingestion is async; give it a moment
time.sleep(2)

try:
    fetched = lf.fetch_trace(trace_id)
    trace_data = getattr(fetched, "data", fetched)
    _ok("Trace read back", f"name={getattr(trace_data, 'name', '?')!r}  id={getattr(trace_data, 'id', trace_id)}")
except Exception as exc:
    _warn("Could not read trace back", str(exc))
    print("       (Write succeeded — read-back failure may be a timing issue)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n─────────────────────────────────────────")
print(f"{PASS}  Langfuse is reachable and credentials are valid.")
if project_id:
    trace_url = f"{host.rstrip('/')}/project/{project_id}/traces/{trace_id}"
    print(f"     Test trace: {trace_url}")
else:
    print(f"     Set LANGFUSE_PROJECT_ID in .env to get a direct trace URL.")
print("─────────────────────────────────────────\n")
