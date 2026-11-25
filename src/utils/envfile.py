"""Loads the `.env` file (API keys) into the process environment.

The `.env` file lives next to `cli.py`/`menu.py`. Format: `NAME=value`,
one line per key; blank lines and lines starting with `#` are ignored.
Handles Windows line endings (CRLF), spaces around the `=` sign and quoted
values. Variables already present in the environment are NOT overwritten
(so a value set by hand takes precedence).
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path_arg: str | os.PathLike | None = None) -> int:
    """Loads the .env file into os.environ. Returns how many keys were loaded."""
    path = Path(path_arg) if path_arg else Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return 0
    loaded = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if not name or not value:
            continue
        if name not in os.environ:
            os.environ[name] = value
            loaded += 1
    return loaded
