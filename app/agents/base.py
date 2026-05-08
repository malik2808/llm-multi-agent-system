from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy.orm import Session

from app.context import ContextBudgetManager, SharedContext
from app.logging_utils import Timer, log_event
from app.models import AgentRun
from app.prompts import get_prompt
from app.schemas import AgentOutput
from app.tools import ToolExecutor
from app.utils import estimate_tokens


class BaseAgent(ABC):
    agent_id: str
    prompt_name: str
    default_budget: int

    def __init__(self, *, max_context_budget: int | None = None) -> None:
        self.max_context_budget = max_context_budget or self.default_budget

    def execute(
        self,
        session: Session,
        *,
        context: SharedContext,
        budget_manager: ContextBudgetManager,
        compression_agent: Any,
        tool_executor: ToolExecutor,
        run_key: str | None = None,
        **kwargs: Any,
    ) -> AgentOutput:
        run_key = run_key or self.agent_id
        budget_manager.declare_budget(run_key, self.max_context_budget)
        prompt = get_prompt(session, self.prompt_name)
        assembled_context = budget_manager.enforce_before_run(context, run_key, compression_agent)
        remaining_before = budget_manager.remaining(run_key)
        log_event(
            session,
            job_id=context.job_id,
            agent_id=run_key,
            event_type="agent_started",
            payload={
                "prompt_name": self.prompt_name,
                "prompt_version": prompt.version,
                "max_context_budget": self.max_context_budget,
                "budget_remaining": remaining_before,
            },
            input_value=assembled_context,
            token_count=estimate_tokens(assembled_context),
        )
        session.commit()

        with Timer() as timer:
            output = self._run(
                session=session,
                context=context,
                context_input=assembled_context,
                prompt_text=prompt.text,
                tool_executor=tool_executor,
                run_key=run_key,
                **kwargs,
            )
        output.token_count = estimate_tokens(output.model_dump())
        violations = budget_manager.add_usage(run_key, output.model_dump())
        session.add(
            AgentRun(
                job_id=context.job_id,
                agent_id=run_key,
                prompt_name=self.prompt_name,
                prompt_version=prompt.version,
                prompt_text=prompt.text,
                input=assembled_context,
                output=output.model_dump(),
                token_count=output.token_count,
                latency_ms=timer.latency_ms,
            )
        )
        context.add_agent_output(run_key, output.model_dump())
        self._stream_text(session, context.job_id, run_key, output.text, budget_manager.remaining(run_key))
        log_event(
            session,
            job_id=context.job_id,
            agent_id=run_key,
            event_type="agent_completed",
            payload={
                "token_count": output.token_count,
                "budget_remaining": budget_manager.remaining(run_key),
                "artifact_keys": sorted(output.artifacts.keys()),
            },
            output_value=output.model_dump(),
            latency_ms=timer.latency_ms,
            token_count=output.token_count,
            policy_violations=violations,
        )
        session.commit()
        return output

    def _stream_text(self, session: Session, job_id: str, agent_id: str, text: str, budget_remaining: int) -> None:
        for index, token in enumerate(text.split()):
            log_event(
                session,
                job_id=job_id,
                agent_id=agent_id,
                event_type="stream_token",
                payload={"token": token, "sequence": index, "budget_remaining": budget_remaining},
                output_value=token,
                token_count=1,
            )
        session.commit()

    @abstractmethod
    def _run(
        self,
        *,
        session: Session,
        context: SharedContext,
        context_input: dict[str, Any],
        prompt_text: str,
        tool_executor: ToolExecutor,
        run_key: str,
        **kwargs: Any,
    ) -> AgentOutput:
        raise NotImplementedError

