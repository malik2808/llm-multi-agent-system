from __future__ import annotations

import hashlib
import json
import re
from typing import Any


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, default=str)
    return max(1, len(TOKEN_RE.findall(value)))


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str) if not isinstance(value, str) else value
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sentence_split(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def redact_large(value: Any, limit: int = 8000) -> Any:
    serialized = json.dumps(value, sort_keys=True, default=str)
    if len(serialized) <= limit:
        return value
    return {"truncated": True, "hash": stable_hash(value), "preview": serialized[:limit]}

