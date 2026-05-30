"""Debug utilities for qrk_adv."""

DEBUG = False


def set_debug(enabled: bool) -> None:
    """Enable or disable debug logging for qrk_adv."""
    global DEBUG
    DEBUG = bool(enabled)


def debug_log(msg: str) -> None:
    """Print debug messages when DEBUG is enabled."""
    if DEBUG:
        print(msg)
