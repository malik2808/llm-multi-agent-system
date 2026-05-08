from __future__ import annotations

from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.models import EventLog, ToolCallLog
from app.utils import estimate_tokens, stable_hash


def log_event(
    session: Session,
    *,
    job_id: str | None,
    agent_id: str | None,
    event_type: str,
    payload: dict[str, Any] | None = None,
    input_value: Any | None = None,
    output_value: Any | None = None,
    latency_ms: float | None = None,
    token_count: int | None = None,
    policy_violations: list[dict[str, Any]] | None = None,
) -> EventLog:
    event = EventLog(
        job_id=job_id,
        agent_id=agent_id,
        event_type=event_type,
        payload=payload or {},
        input_hash=stable_hash(input_value) if input_value is not None else None,
        output_hash=stable_hash(output_value) if output_value is not None else None,
        latency_ms=latency_ms,
        token_count=token_count if token_count is not None else estimate_tokens(output_value),
        policy_violations=policy_violations or [],
    )
    session.add(event)
    session.flush()
    return event


def log_tool_call(
    session: Session,
    *,
    job_id: str,
    agent_id: str,
    tool_name: str,
    attempt: int,
    input_value: dict[str, Any],
    output_value: dict[str, Any],
    latency_ms: float,
    accepted: bool,
    rejection_reason: str | None,
) -> ToolCallLog:
    row = ToolCallLog(
        job_id=job_id,
        agent_id=agent_id,
        tool_name=tool_name,
        attempt=attempt,
        input=input_value,
        output=output_value,
        latency_ms=latency_ms,
        accepted=accepted,
        rejection_reason=rejection_reason,
    )
    session.add(row)
    session.flush()
    return row


class Timer:
    def __enter__(self) -> "Timer":
        self.started = perf_counter()
        self.latency_ms = 0.0
        return self

    def __exit__(self, *_args: object) -> None:
        self.latency_ms = (perf_counter() - self.started) * 1000

