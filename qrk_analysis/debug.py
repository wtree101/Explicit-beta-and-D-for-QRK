"""Opt-in diagnostics for numerical searches."""

_DEBUG_ENABLED = False


def set_debug(enabled: bool) -> None:
    """Enable or disable diagnostic output from the analysis package."""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = bool(enabled)


def debug_log(message: str) -> None:
    """Print ``message`` only when diagnostics are enabled."""
    if _DEBUG_ENABLED:
        print(message)
