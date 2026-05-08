from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select

from app.db import SessionLocal, init_db, session_scope
from app.eval_harness import EvalHarness
from app.models import AgentRun, EvalRun, EventLog, Job, PromptRewrite, ToolCallLog, utc_now
from app.prompts import approve_prompt
from app.schemas import (
    ErrorResponse,
    EvalSummaryResponse,
    PromptDecisionRequest,
    PromptDecisionResponse,
    QueryRequest,
    RerunFailedResponse,
    TraceResponse,
)
from app.seed import seed_reference_data


app = FastAPI(
    title="LLM Engineer Assessment API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.on_event("startup")
def startup() -> None:
    init_db()
    with session_scope() as session:
        seed_reference_data(session)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"error_code": "http_error", "message": str(exc.detail), "job_id": None}
    return JSONResponse(status_code=exc.status_code, content=detail)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error_code="validation_error", message=str(exc), job_id=None).model_dump(),
    )


@app.post("/query")
async def submit_query(request: QueryRequest) -> StreamingResponse:
    """Submit a user query and stream real-time agent/tool activity as Server-Sent Events."""
    with session_scope() as session:
        job = Job(query=request.query, status="queued")
        session.add(job)
        session.flush()
        job_id = job.id

    async def event_stream() -> Any:
        yield sse("job_queued", {"job_id": job_id, "status": "queued"}, event_id="0")
        last_id = 0
        idle_ticks = 0
        while True:
            with SessionLocal() as session:
                events = session.scalars(
                    select(EventLog)
                    .where(EventLog.job_id == job_id, EventLog.id > last_id)
                    .order_by(EventLog.id.asc())
                    .limit(100)
                ).all()
                job = session.get(Job, job_id)
                for event in events:
                    last_id = event.id
                    yield sse(event.event_type, serialize_event(event), event_id=str(event.id))
                if job and job.status in {"completed", "failed"} and not events:
                    yield sse(
                        "job_terminal",
                        {
                            "job_id": job_id,
                            "status": job.status,
                            "final_answer": job.final_answer,
                            "error_code": job.error_code,
                            "message": job.error_message,
                        },
                    )
                    break
            if not events:
                idle_ticks += 1
                if idle_ticks > 600:
                    yield sse("stream_timeout", {"job_id": job_id, "status": "timeout_waiting_for_worker"})
                    break
            else:
                idle_ticks = 0
            await asyncio.sleep(0.25)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/trace/{job_id}", response_model=TraceResponse)
def get_trace(job_id: str) -> TraceResponse:
    """Return the complete ordered execution trace for a completed or in-flight job."""
    with session_scope() as session:
        job = session.get(Job, job_id)
        if not job:
            raise api_error(404, "job_not_found", f"No job exists for id {job_id}", job_id)
        events = session.scalars(select(EventLog).where(EventLog.job_id == job_id).order_by(EventLog.id.asc())).all()
        tool_calls = session.scalars(select(ToolCallLog).where(ToolCallLog.job_id == job_id).order_by(ToolCallLog.id.asc())).all()
        agent_runs = session.scalars(select(AgentRun).where(AgentRun.job_id == job_id).order_by(AgentRun.id.asc())).all()
        return TraceResponse(
            job_id=job.id,
            status=job.status,
            query=job.query,
            final_answer=job.final_answer,
            events=[serialize_event(event) for event in events],
            tool_calls=[serialize_tool_call(call) for call in tool_calls],
            agent_runs=[serialize_agent_run(run) for run in agent_runs],
        )


@app.get("/eval/latest", response_model=EvalSummaryResponse)
def latest_eval_summary() -> EvalSummaryResponse:
    """Return the most recent evaluation run summary by category and scoring dimension."""
    with session_scope() as session:
        eval_run = session.scalars(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(1)).first()
        if not eval_run:
            raise api_error(404, "eval_not_found", "No evaluation run has been recorded yet")
        return EvalSummaryResponse(
            eval_run_id=eval_run.id,
            created_at=eval_run.created_at,
            status=eval_run.status,
            targeted=eval_run.targeted,
            summary=eval_run.summary,
        )


@app.post("/prompt-rewrites/{rewrite_id}/decision", response_model=PromptDecisionResponse)
def decide_prompt_rewrite(rewrite_id: str, request: PromptDecisionRequest) -> PromptDecisionResponse:
    """Approve or reject a pending prompt rewrite proposed by the meta-agent."""
    with session_scope() as session:
        rewrite = session.get(PromptRewrite, rewrite_id)
        if not rewrite:
            raise api_error(404, "rewrite_not_found", f"No prompt rewrite exists for id {rewrite_id}")
        if rewrite.status != "pending":
            raise api_error(409, "rewrite_already_decided", f"Rewrite {rewrite_id} is already {rewrite.status}")
        approved_version = None
        if request.decision == "approve":
            prompt = approve_prompt(session, rewrite.prompt_name, rewrite.proposed_text)
            approved_version = prompt.version
            rewrite.status = "approved"
        else:
            rewrite.status = "rejected"
        rewrite.decision_reason = request.reason
        rewrite.decided_at = utc_now()
        session.flush()
        return PromptDecisionResponse(
            rewrite_id=rewrite.id,
            status=rewrite.status,
            prompt_name=rewrite.prompt_name,
            approved_version=approved_version,
        )


@app.post("/eval/rerun-failed", response_model=RerunFailedResponse)
def rerun_failed_cases() -> RerunFailedResponse:
    """Run a targeted evaluation on the latest failed cases using the latest approved prompts."""
    with session_scope() as session:
        harness = EvalHarness()
        source_eval_run_id, failed_case_ids = harness.latest_failed_case_ids(session)
        if not failed_case_ids:
            raise api_error(409, "no_failed_cases", "The latest evaluation run has no failed cases to rerun")
        eval_run = harness.run(
            session,
            case_ids=failed_case_ids,
            targeted=True,
            source_eval_run_id=source_eval_run_id,
        )
        return RerunFailedResponse(
            eval_run_id=eval_run.id,
            source_eval_run_id=source_eval_run_id,
            failed_cases=len(failed_case_ids),
            summary=eval_run.summary,
        )


def api_error(status_code: int, error_code: str, message: str, job_id: str | None = None) -> HTTPException:
    return HTTPException(status_code=status_code, detail=ErrorResponse(error_code=error_code, message=message, job_id=job_id).model_dump())


def sse(event: str, data: dict[str, Any], event_id: str | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, default=str)}")
    return "\n".join(lines) + "\n\n"


def serialize_event(event: EventLog) -> dict[str, Any]:
    return {
        "id": event.id,
        "timestamp": event.timestamp.isoformat(),
        "job_id": event.job_id,
        "agent_id": event.agent_id,
        "event_type": event.event_type,
        "input_hash": event.input_hash,
        "output_hash": event.output_hash,
        "latency_ms": event.latency_ms,
        "token_count": event.token_count,
        "policy_violations": event.policy_violations,
        "payload": event.payload,
    }


def serialize_tool_call(call: ToolCallLog) -> dict[str, Any]:
    return {
        "id": call.id,
        "timestamp": call.timestamp.isoformat(),
        "job_id": call.job_id,
        "agent_id": call.agent_id,
        "tool_name": call.tool_name,
        "attempt": call.attempt,
        "input": call.input,
        "output": call.output,
        "latency_ms": call.latency_ms,
        "accepted": call.accepted,
        "rejection_reason": call.rejection_reason,
    }


def serialize_agent_run(run: AgentRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "timestamp": run.created_at.isoformat(),
        "job_id": run.job_id,
        "agent_id": run.agent_id,
        "prompt_name": run.prompt_name,
        "prompt_version": run.prompt_version,
        "prompt_text": run.prompt_text,
        "input": run.input,
        "output": run.output,
        "token_count": run.token_count,
        "latency_ms": run.latency_ms,
    }

