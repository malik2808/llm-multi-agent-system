from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.schemas import ToolResult, ToolStatus


DISALLOWED = ("import os", "import subprocess", "socket", "open(", "__import__", "eval(", "exec(")


class CodeExecutionSandbox:
    name = "code_execution_sandbox"

    def run(self, session: Session, payload: dict[str, Any]) -> ToolResult:
        del session
        code = str(payload.get("code", "")).strip()
        timeout_ms = int(payload.get("timeout_ms", 1000))
        if not code:
            return ToolResult(status=ToolStatus.malformed, error_code="code_empty", message="code is required")
        if len(code) > 2000:
            return ToolResult(status=ToolStatus.malformed, error_code="code_too_long", message="code exceeds 2000 characters")
        lowered = code.lower()
        if any(token in lowered for token in DISALLOWED):
            return ToolResult(status=ToolStatus.malformed, error_code="code_disallowed", message="code contains disallowed operations")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snippet.py"
            path.write_text(code, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(path)],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=timeout_ms / 1000,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return ToolResult(
                    status=ToolStatus.timeout,
                    error_code="code_timeout",
                    message="Python snippet exceeded timeout",
                    payload={"stdout": exc.stdout or "", "stderr": exc.stderr or "", "exit_code": None},
                )
        return ToolResult(
            status=ToolStatus.ok,
            payload={
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "exit_code": completed.returncode,
            },
            message="snippet executed",
        )

