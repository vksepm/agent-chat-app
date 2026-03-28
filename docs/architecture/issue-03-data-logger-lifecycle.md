# Issue 03 — Deepen `data_logger.py` Global State into a `DataLogger` Class

## Problem

`src/data_logger.py` exposes a module-level API built on seven global mutable
variables (lines 228–234):

```python
_log_queue:    Queue               = Queue()
_jsonl_path:   Path | None         = None
_repo_id:      str | None          = None
_hf_token:     str | None          = None
_sync_interval: int                = 300
_sync_stop:    threading.Event     = threading.Event()
_write_lock:   threading.Lock      = threading.Lock()
```

`init_logger()` (lines 273–319) mutates these globals and spawns two daemon threads.
Calling `init_logger()` a second time would spawn duplicate worker threads without any
guard.  There is no `shutdown_logger()` to drain the queue and stop workers cleanly.

**Why this is tightly coupled:**
`app.py` calls `init_logger()` once at module load and then calls `log_interaction()`
at the end of every `chat()` turn.  The implicit contract is that `init_logger()` has
already been called and globals are populated.  If `_jsonl_path` is `None` (e.g.,
`init_logger()` not called, or called after a test reset), `_log_worker()` silently
discards entries (line 242: `if isinstance(entry, ...) and _jsonl_path is not None`).

**Why this makes the codebase hard to test:**
- Tests must manipulate global state to configure the logger, and rely on the
  background queue draining (`_log_queue.join()`) before asserting file contents.
  This is fragile: any test that spawns the worker threads without later stopping
  them leaves background threads running across tests.
- `test_data_logger.py` works around this by using `tmp_path` fixtures but cannot
  reset the global `_jsonl_path` between tests without monkey-patching the module.
- It is impossible to run two logger configurations in the same process (e.g., one
  logger per integration test scenario).

**Integration risk:**
- App shutdown (Gradio `demo.close()` or Ctrl-C) happens before the daemon threads
  drain the queue.  Log entries enqueued at the end of the last `chat()` turn may
  be lost.
- HF Hub sync (`_sync_worker`) runs on an independent timer; there is no way to
  wait for it to complete at shutdown.

## Proposed Interface

Replace the module-level API with a `DataLogger` class that owns its own threads,
queue, and lock:

```python
# src/data_logger.py  (revised public API)

class DataLogger:
    """
    Non-blocking interaction logger with optional HuggingFace Hub sync.

    Usage
    -----
    logger = DataLogger(log_dir="logs", repo_id="user/ds", hf_token="hf_...")
    logger.start()          # spawn worker threads

    logger.log(...)         # non-blocking enqueue

    logger.shutdown(        # drain queue, stop sync, optional final upload
        timeout=10,
        final_sync=True,
    )
    ```

    # Or use as a context manager:
    with DataLogger(...) as dl:
        dl.log(...)
    """

    def __init__(
        self,
        log_dir: str = "logs",
        repo_id: str | None = None,
        hf_token: str | None = None,
        sync_interval: int = 300,
    ) -> None: ...

    def start(self) -> "DataLogger": ...

    def log(
        self,
        conversation_id: str,
        model_id: str,
        app_version: str,
        user_input: str,
        final_answer: str,
        agent_steps: list[dict],
        turn_number: int = 0,
        extra: dict | None = None,
    ) -> None:
        """Non-blocking.  Enqueues entry; returns immediately."""
        ...

    def shutdown(self, timeout: float = 10.0, final_sync: bool = False) -> None:
        """
        Drain the log queue (up to *timeout* seconds), stop sync worker,
        and optionally push a final HF Hub upload.
        """
        ...

    def __enter__(self) -> "DataLogger": ...
    def __exit__(self, *_) -> None: ...
```

**`app.py` usage after migration:**

```python
from src.data_logger import DataLogger

_data_logger = DataLogger(
    log_dir=cfg.data_log_dir,
    repo_id=cfg.hf_dataset_repo_id,
    hf_token=cfg.hf_token,
    sync_interval=cfg.hf_sync_interval,
).start()

# Register shutdown hook so the queue drains on Ctrl-C or demo.close():
import atexit
atexit.register(_data_logger.shutdown, timeout=10, final_sync=True)

# In chat():
_data_logger.log(...)
```

What the class hides:
- Thread lifecycle (start, stop, daemon flags).
- Global variable mutations.
- `_log_queue.join()` drain logic in `shutdown()`.
- Duplicate-init protection (raise `RuntimeError` if `start()` called twice).

## Dependency Strategy

**Remote but owned (Ports & Adapters)** — HuggingFace Hub is your own infrastructure.

- Production: `DataLogger` uploads via `huggingface_hub.HfApi`.
- Testing: Inject a mock hub upload callable via a `_upload_fn` parameter (or
  `unittest.mock.patch`).  The local JSONL write path uses `tmp_path` (stdlib,
  no network).

```python
# Constructor addition for testing:
def __init__(
    self,
    ...,
    _upload_fn=None,   # Injected in tests; defaults to real HfApi upload
) -> None:
    self._upload_fn = _upload_fn or _default_hf_upload
```

Alternatively, patch `_upload_to_hub` at the class boundary.  Either approach ensures
the threading model, queue draining, and JSONL serialisation are all tested without
hitting HF Hub.

## Testing Strategy

**New boundary tests to write (`tests/unit/test_data_logger.py` — additions):**

- `test_start_spawns_worker_threads` — after `start()`, assert two threads with
  names `data-log-worker` / `data-sync-worker` are alive.
- `test_shutdown_drains_queue` — enqueue 5 log entries, call `shutdown(timeout=5)`,
  assert all entries written to JSONL file.
- `test_double_start_raises` — calling `start()` twice raises `RuntimeError`.
- `test_log_after_shutdown_is_silently_dropped` — calling `log()` after `shutdown()`
  does not raise and does not block.
- `test_context_manager_calls_shutdown` — use `with DataLogger(...) as dl: dl.log(...)`;
  assert file written after `with` block exits.
- `test_hf_sync_disabled_when_no_credentials` — instantiate with no `repo_id`;
  assert only one worker thread spawned (no sync thread).
- `test_shutdown_calls_final_sync_when_requested` — mock upload function; assert
  called exactly once when `shutdown(final_sync=True)`.
- `test_invalid_log_entry_does_not_crash` — enqueue an entry with an empty
  `user_input` (Pydantic validation failure); assert no exception propagates and
  JSONL file contains no garbage.

**Old tests to update:**
- Existing tests in `test_data_logger.py` that call the module-level `init_logger()`
  / `log_interaction()` must be rewritten to instantiate `DataLogger` directly.
- The `_log_queue.join()` drain pattern in existing tests is replaced by
  `shutdown(timeout=5)`.

**Test environment needs:**
- `tmp_path` pytest fixture (stdlib, no install).
- Mock upload callable — `unittest.mock.MagicMock()`.
- No HuggingFace credentials or network access.

## Implementation Recommendations

**What the class should own:**
- All thread lifecycle management (start, stop, daemon flags, join).
- Queue, write lock, stop event — all as instance attributes, not module globals.
- Duplicate-init guard (raise `RuntimeError` if `start()` called twice on same
  instance).
- `shutdown(timeout, final_sync)` drain/stop/sync sequence.

**What it should hide:**
- Global variable mutation.
- The distinction between "log thread" and "sync thread" from callers.
- JSONL path construction and file locking.

**What it should expose:**
- `DataLogger(log_dir, repo_id, hf_token, sync_interval)` constructor.
- `start()` — returns `self` for chaining.
- `log(...)` — non-blocking enqueue (same parameters as current `log_interaction()`).
- `shutdown(timeout, final_sync)` — blocking drain + optional upload.
- Context manager protocol (`__enter__` / `__exit__`).

**How callers should migrate:**
1. Add `DataLogger` class to `src/data_logger.py` (or rename to
   `src/data_logger_v2.py` during transition).
2. Retain module-level `init_logger()` / `log_interaction()` as thin wrappers around
   a module-level `DataLogger` instance until all callers migrate.
3. Update `app.py` to instantiate `DataLogger` directly and register `atexit` hook.
4. Update all tests to use `DataLogger` directly.
5. Delete the module-level globals and wrapper functions once no caller uses them.
