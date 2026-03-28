"""
src/timer.py — Lightweight step timer utility.

Records wall-clock duration for named checkpoints so callers can report
detailed timing breakdowns in log entries.
"""

import json
import time


class Timer:
    """
    Multi-step wall-clock timer.

    Usage
    -----
    >>> t = Timer("my_operation")
    >>> t.start()
    >>> do_work()
    >>> t.add_step("step 1")
    >>> do_more_work()
    >>> t.add_step("step 2")
    >>> t.end()
    >>> print(t.formatted_result())
    """

    def __init__(self, name: str | None = None):
        self.name = name
        self.start_time: float | None = None
        self.steps: list[dict] = []
        self.total_time: float | None = None

    def clear(self) -> None:
        self.start_time = None
        self.steps = []
        self.total_time = None

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()

    def is_running(self) -> bool:
        return self.start_time is not None

    def add_step(self, step_name: str) -> None:
        """Record a named checkpoint with elapsed time since the last step (or start)."""
        if self.start_time is None:
            self.start()
        current = time.time()
        elapsed = (
            current - self.steps[-1]["timestamp"]
            if self.steps
            else current - self.start_time
        )
        self.steps.append(
            {
                "step_name": step_name,
                "duration": round(elapsed, 4),
                "total_duration": round(current - self.start_time, 4),
                "timestamp": current,
            }
        )

    def end(self) -> None:
        """Finalise the timer. Must be called after at least one add_step()."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")
        if not self.steps:
            raise RuntimeError("No steps have been added.")
        self.total_time = time.time() - self.start_time

    def to_json(self) -> dict:
        """Return a compact dict with name, total_time, and per-step durations."""
        if self.total_time is None:
            raise RuntimeError("Timer has not been ended.")
        output: dict = {}
        if self.name:
            output["name"] = self.name
        output["total_time"] = round(self.total_time, 4)
        for step in self.steps:
            output[step["step_name"]] = step["duration"]
        return output

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=4)

    def formatted_result(self) -> str:
        """Return a human-readable multi-line summary."""
        if self.total_time is None:
            raise RuntimeError("Timer has not been ended.")
        lines: list[str] = []
        if self.name:
            lines.append(f"Timer: {self.name}")
        for step in self.steps:
            lines.append(
                f"[{step['duration']:05.2f}s, {step['total_duration']:05.2f}s] {step['step_name']}"
            )
        lines.append(f"Total time: {self.total_time:.2f}s")
        return "\n".join(lines)
