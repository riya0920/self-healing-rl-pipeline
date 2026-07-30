"""
Self-Healing RL Recommendation Agent — Observability (LangSmith)

Wraps the multi-agent healing loop with LangSmith tracing so every
Monitor -> Diagnostics -> Repair -> Verification cycle shows up as a
single nested trace (with the MCP tool calls as child runs).

Design goals:
  * Zero hard dependency — if `langsmith` isn't installed, or no
    LANGSMITH_API_KEY is set, every decorator becomes a no-op and the
    pipeline runs exactly as before.
  * One import surface — code elsewhere just does:
        from observability import traceable
    and decorates functions with @traceable(...) without caring whether
    tracing is actually on.

Enable tracing by setting environment variables (see .env.example):
    LANGSMITH_API_KEY=ls__...
    LANGSMITH_PROJECT=self-healing-rl     # optional, defaults below
"""

import os

# ------------------------------------------------------------------
# Decide whether tracing is active
# ------------------------------------------------------------------
# Tracing is "requested" when an API key is present. We also honour an
# explicit LANGSMITH_TRACING=false kill-switch.
_KILL_SWITCH = os.environ.get("LANGSMITH_TRACING", "").lower() in ("false", "0", "no")
_HAS_KEY = bool(os.environ.get("LANGSMITH_API_KEY"))
_REQUESTED = _HAS_KEY and not _KILL_SWITCH

if _REQUESTED:
    # LangSmith's SDK reads these env vars; set sane defaults so the user
    # only strictly needs to provide the API key.
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "self-healing-rl")

# ------------------------------------------------------------------
# Try to import the real LangSmith decorator
# ------------------------------------------------------------------
try:
    from langsmith import traceable as _ls_traceable

    _AVAILABLE = True
except Exception:  # ImportError, or any SDK init problem
    _AVAILABLE = False

TRACING_ENABLED = _REQUESTED and _AVAILABLE


def _wrap(func, *dargs, **dkwargs):
    if TRACING_ENABLED:
        return _ls_traceable(*dargs, **dkwargs)(func)
    return func


def traceable(*dargs, **dkwargs):
    """
    Drop-in stand-in for langsmith.traceable that degrades to a no-op.

    Supports both usages:
        @traceable
        def f(...): ...

        @traceable(run_type="chain", name="monitor.detect_drift")
        def g(...): ...
    """
    # Bare decorator form: @traceable
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return _wrap(dargs[0])

    # Parameterized form: @traceable(...)
    def decorator(func):
        return _wrap(func, *dargs, **dkwargs)

    return decorator


def status_line() -> str:
    """Human-readable one-liner describing the current tracing state."""
    if TRACING_ENABLED:
        project = os.environ.get("LANGSMITH_PROJECT", "self-healing-rl")
        return f"🔭 LangSmith tracing ON  (project: {project})"
    if _REQUESTED and not _AVAILABLE:
        return "🔭 LangSmith requested but SDK not installed  (pip install langsmith)"
    return "🔭 LangSmith tracing OFF  (set LANGSMITH_API_KEY to enable)"
