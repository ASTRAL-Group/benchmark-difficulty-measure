"""Health logging for mass-fill (heartbeat + human-readable log)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mass_fill.distribution import MassFillState, results_dir


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def log_path(benchmark: str, root: Path | None = None) -> Path:
    return results_dir(benchmark, root) / "logs" / "mass_fill.log"


def heartbeat_path(benchmark: str, root: Path | None = None) -> Path:
    return results_dir(benchmark, root) / "logs" / "heartbeat.json"


def append_log(benchmark: str, message: str, root: Path | None = None) -> None:
    path = log_path(benchmark, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{_now()}] {message}"
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def write_heartbeat(
    state: MassFillState,
    *,
    status: str,
    phase_detail: str = "",
    last_error: str = "",
    extra: dict[str, Any] | None = None,
    root: Path | None = None,
) -> None:
    path = heartbeat_path(state.benchmark, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": status,
        "benchmark": state.benchmark,
        "filled": state.filled_count(),
        "target_n": state.target_count,
        "remaining": state.remaining(),
        "pct": round(100.0 * state.filled_count() / max(1, state.target_count), 1),
        "seed": state.seed_count(),
        "real": state.real_count(),
        "pretend": state.pretend_count(),
        "step": state.step,
        "completed": state.completed,
        "phase_detail": phase_detail,
        "last_update": _now(),
        "unix_time": time.time(),
        "last_error": last_error,
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def print_status(benchmark: str, root: Path | None = None) -> None:
    hb = heartbeat_path(benchmark, root)
    log = log_path(benchmark, root)
    print(f"=== mass-fill status: {benchmark} ===")
    if hb.is_file():
        print(hb.read_text(encoding="utf-8"))
    else:
        print("(no heartbeat.json yet)")
    print("--- last log lines ---")
    if log.is_file():
        lines = log.read_text(encoding="utf-8").splitlines()
        for line in lines[-8:]:
            print(line)
    else:
        print("(no mass_fill.log yet)")
