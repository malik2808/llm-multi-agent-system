from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy.orm import Session

from app.logging_utils import Timer, log_event, log_tool_call
from app.schemas import ToolResult, ToolStatus


class BaseTool(ABC):
    name: str

    @abstractmethod
    def run(self, session: Session, payload: dict[str, Any]) -> ToolResult:
        raise NotImplementedError


class ToolExecutor:
    def __init__(self, tools: list[BaseTool]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def call(
        self,
        session: Session,
        *,
        job_id: str,
        agent_id: str,
        tool_name: str,
        payload: dict[str, Any],
        max_retries: int = 2,
        accept_empty: bool = False,
    ) -> ToolResult:
        tool = self.tools[tool_name]
        current_payload = dict(payload)
        result: ToolResult | None = None
        for attempt in range(1, max_retries + 2):
            with Timer() as timer:
                result = tool.run(session, current_payload)
            result.latency_ms = timer.latency_ms
            accepted, reason = self._acceptance_decision(result, accept_empty=accept_empty)
            output = result.model_dump()
            log_tool_call(
                session,
                job_id=job_id,
                agent_id=agent_id,
                tool_name=tool_name,
                attempt=attempt,
                input_value=current_payload,
                output_value=output,
                latency_ms=timer.latency_ms,
                accepted=accepted,
                rejection_reason=None if accepted else reason,
            )
            log_event(
                session,
                job_id=job_id,
                agent_id=agent_id,
                event_type="tool_call",
                payload={
                    "tool_name": tool_name,
                    "attempt": attempt,
                    "status": result.status.value,
                    "accepted": accepted,
                    "rejection_reason": None if accepted else reason,
                },
                input_value=current_payload,
                output_value=output,
                latency_ms=timer.latency_ms,
            )
            session.commit()
            if accepted:
                return result
            if attempt > max_retries:
                return result
            current_payload = self._retry_payload(result, current_payload, attempt)
        if result is None:
            return ToolResult(status=ToolStatus.error, error_code="tool_not_invoked", message="Tool was not invoked")
        return result

    def _acceptance_decision(self, result: ToolResult, *, accept_empty: bool) -> tuple[bool, str]:
        if result.status == ToolStatus.ok:
            return True, ""
        if result.status == ToolStatus.empty and accept_empty:
            return True, ""
        if result.status == ToolStatus.timeout:
            return False, "timeout: retry with a relaxed or narrower input"
        if result.status == ToolStatus.empty:
            return False, "empty: retry with broadened input or fallback"
        if result.status == ToolStatus.malformed:
            return False, "malformed: retry with sanitized input"
        return False, result.message or "tool error"

    def _retry_payload(self, result: ToolResult, payload: dict[str, Any], attempt: int) -> dict[str, Any]:
        retry = dict(payload)
        retry["retry_attempt"] = attempt
        if result.status == ToolStatus.timeout:
            retry["timeout_ms"] = int(retry.get("timeout_ms", 1000)) * 2
            retry["query"] = retry.get("query") or retry.get("question") or ""
        elif result.status == ToolStatus.empty:
            if "query" in retry:
                retry["query"] = f"{retry['query']} overview facts context"
            if "question" in retry:
                retry["question"] = f"{retry['question']} known fact"
        elif result.status == ToolStatus.malformed:
            for key in ("query", "question", "code"):
                if key in retry and isinstance(retry[key], str):
                    retry[key] = retry[key].strip()[:2000]
        return retry

