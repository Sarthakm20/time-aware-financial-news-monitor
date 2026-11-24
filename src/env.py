"""Read API keys from the environment or a git-ignored ``.env`` file in the project root.

``.env`` format (see ``.env.example``)::

    GEMINI_API_KEY=your-key
    FINNHUB_API_KEY=

Real environment variables win over ``.env``. Keys are never printed or logged.
"""

import os
from pathlib import Path
from typing import Dict, Optional

from src.config import PROJECT_ROOT

ENV_FILE = PROJECT_ROOT / ".env"


def read_env_file(path: Path = ENV_FILE) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_env(path: Path = ENV_FILE) -> None:
    """Copy ``.env`` values into ``os.environ`` without overriding variables that are already set."""
    for key, value in read_env_file(path).items():
        if value and not os.environ.get(key):
            os.environ[key] = value


def get_secret(name: str, path: Path = ENV_FILE) -> Optional[str]:
    value = os.environ.get(name, "").strip() or read_env_file(path).get(name, "").strip()
    return value or None
