from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    job_id: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class TraceResponse(BaseModel):
    job_id: str
    status: str
    query: str
    final_answer: str | None
    events: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    agent_runs: list[dict[str, Any]]


class EvalSummaryResponse(BaseModel):
    eval_run_id: str
    created_at: datetime
    status: str
    targeted: bool
    summary: dict[str, Any]


class PromptDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reason: str = Field(min_length=1, max_length=2000)


class PromptDecisionResponse(BaseModel):
    rewrite_id: str
    status: str
    prompt_name: str
    approved_version: int | None = None


class RerunFailedResponse(BaseModel):
    eval_run_id: str
    source_eval_run_id: str
    failed_cases: int
    summary: dict[str, Any]


class ToolStatus(str, Enum):
    ok = "ok"
    timeout = "timeout"
    empty = "empty"
    malformed = "malformed"
    error = "error"


class ToolResult(BaseModel):
    status: ToolStatus
    payload: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    message: str = ""
    latency_ms: float = 0.0


class AgentOutput(BaseModel):
    agent_id: str
    claims: list[dict[str, Any]] = Field(default_factory=list)
    text: str = ""
    artifacts: dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0


class RoutingDecision(BaseModel):
    next_agent: str
    reason: str
    context_budget: int
    depends_on: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)

