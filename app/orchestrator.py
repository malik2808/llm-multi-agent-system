from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.agents import CompressionAgent, CritiqueAgent, DecompositionAgent, RetrievalAgent, SynthesisAgent
from app.context import ContextBudgetManager, SharedContext
from app.logging_utils import log_event
from app.models import Job
from app.schemas import RoutingDecision
from app.tools import CodeExecutionSandbox, SelfReflectionTool, StructuredDataLookupTool, ToolExecutor, WebSearchStub


class Orchestrator:
    def __init__(self) -> None:
        self.compression_agent = CompressionAgent()
        self.decomposition_agent = DecompositionAgent()
        self.retrieval_agent = RetrievalAgent()
        self.critique_agent = CritiqueAgent()
        self.synthesis_agent = SynthesisAgent()
        self.tool_executor = ToolExecutor(
            [
                WebSearchStub(),
                CodeExecutionSandbox(),
                StructuredDataLookupTool(),
                SelfReflectionTool(),
            ]
        )

    def run_job(self, session: Session, job: Job) -> SharedContext:
        context = SharedContext(job_id=job.id, query=job.query)
        budget_manager = ContextBudgetManager()
        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        self._event(session, context, "job_started", {"query": job.query})

        try:
            self._route(
                session,
                context,
                RoutingDecision(
                    next_agent="decomposition",
                    reason="No task graph exists yet; first build typed subtasks and dependency gates.",
                    context_budget=self.decomposition_agent.max_context_budget,
                ),
            )
            self.decomposition_agent.execute(
                session,
                context=context,
                budget_manager=budget_manager,
                compression_agent=self.compression_agent,
                tool_executor=self.tool_executor,
                run_key="decomposition",
            )

            retrieval_round = 1
            while self._pending_tasks(context):
                executable = self._executable_tasks(context)
                if not executable:
                    raise RuntimeError("dependency_deadlock: pending subtasks have unresolved dependencies")
                dynamic_budget = max(700, self.retrieval_agent.max_context_budget - (len(context.tool_results) * 25))
                self.retrieval_agent.max_context_budget = dynamic_budget
                self._route(
                    session,
                    context,
                    RoutingDecision(
                        next_agent="retrieval",
                        reason=(
                            "Selected executable subtasks whose dependencies are resolved: "
                            + ", ".join(executable)
                        ),
                        context_budget=dynamic_budget,
                        depends_on=sorted({dep for task_id in executable for dep in context.tasks[task_id]["dependencies"]}),
                        task_ids=executable,
                    ),
                )
                self.retrieval_agent.execute(
                    session,
                    context=context,
                    budget_manager=budget_manager,
                    compression_agent=self.compression_agent,
                    tool_executor=self.tool_executor,
                    run_key=f"retrieval_{retrieval_round}",
                    task_ids=executable,
                )
                retrieval_round += 1

            targets = [key for key in context.agent_outputs if not key.startswith("critique")]
            self._route(
                session,
                context,
                RoutingDecision(
                    next_agent="critique",
                    reason="All executable work is resolved; critique must review each prior agent output at span level.",
                    context_budget=self.critique_agent.max_context_budget,
                    depends_on=targets,
                ),
            )
            self.critique_agent.execute(
                session,
                context=context,
                budget_manager=budget_manager,
                compression_agent=self.compression_agent,
                tool_executor=self.tool_executor,
                run_key="critique_initial",
                target_keys=targets,
            )

            self._route(
                session,
                context,
                RoutingDecision(
                    next_agent="synthesis",
                    reason="Critique feedback is available; synthesize only accepted claims and attach provenance.",
                    context_budget=self.synthesis_agent.max_context_budget,
                    depends_on=["critique_initial"],
                ),
            )
            self.synthesis_agent.execute(
                session,
                context=context,
                budget_manager=budget_manager,
                compression_agent=self.compression_agent,
                tool_executor=self.tool_executor,
                run_key="synthesis_initial",
            )

            self._route(
                session,
                context,
                RoutingDecision(
                    next_agent="critique",
                    reason="Final synthesis is another agent output and must be checked before completion.",
                    context_budget=self.critique_agent.max_context_budget,
                    depends_on=["synthesis_initial"],
                ),
            )
            self.critique_agent.execute(
                session,
                context=context,
                budget_manager=budget_manager,
                compression_agent=self.compression_agent,
                tool_executor=self.tool_executor,
                run_key="critique_synthesis",
                target_keys=["synthesis_initial"],
            )

            synthesis_flags = [
                flag for flag in context.critiques.get("synthesis_initial", []) if not flag.get("agree", True)
            ]
            if synthesis_flags:
                self._route(
                    session,
                    context,
                    RoutingDecision(
                        next_agent="synthesis",
                        reason="Critique disagreed with synthesized spans; rerun synthesis with disputed spans blocked.",
                        context_budget=self.synthesis_agent.max_context_budget,
                        depends_on=["critique_synthesis"],
                    ),
                )
                self.synthesis_agent.execute(
                    session,
                    context=context,
                    budget_manager=budget_manager,
                    compression_agent=self.compression_agent,
                    tool_executor=self.tool_executor,
                    run_key="synthesis_revision",
                    revision=True,
                )

            job.final_answer = context.final_answer
            job.status = "completed"
            job.updated_at = datetime.now(timezone.utc)
            self._event(
                session,
                context,
                "job_completed",
                {"final_answer": context.final_answer, "provenance_map": context.provenance_map},
                output_value=context.final_answer,
            )
            session.commit()
            return context
        except Exception as exc:
            job.status = "failed"
            job.error_code = "orchestration_failed"
            job.error_message = str(exc)
            job.updated_at = datetime.now(timezone.utc)
            self._event(
                session,
                context,
                "job_failed",
                {"error_code": job.error_code, "message": job.error_message},
                policy_violations=[{"type": "runtime_error", "message": str(exc)}],
            )
            session.commit()
            raise

    def _pending_tasks(self, context: SharedContext) -> list[str]:
        return [task_id for task_id, task in context.tasks.items() if task.get("status") == "pending"]

    def _executable_tasks(self, context: SharedContext) -> list[str]:
        executable = []
        for task_id, task in context.tasks.items():
            if task.get("status") != "pending":
                continue
            deps = task.get("dependencies", [])
            if all(context.tasks.get(dep, {}).get("status") == "resolved" for dep in deps):
                executable.append(task_id)
        return executable

    def _route(self, session: Session, context: SharedContext, decision: RoutingDecision) -> None:
        payload = decision.model_dump()
        context.routing_decisions.append(payload)
        context.add_item("routing_decision", payload, agent_id="orchestrator", structured=True)
        self._event(session, context, "routing_decision", payload, agent_id="orchestrator")

    def _event(
        self,
        session: Session,
        context: SharedContext,
        event_type: str,
        payload: dict[str, Any],
        *,
        agent_id: str | None = "orchestrator",
        output_value: Any | None = None,
        policy_violations: list[dict[str, Any]] | None = None,
    ) -> None:
        log_event(
            session,
            job_id=context.job_id,
            agent_id=agent_id,
            event_type=event_type,
            payload=payload,
            output_value=output_value,
            policy_violations=policy_violations or [],
        )
        session.commit()

