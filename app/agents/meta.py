from __future__ import annotations

from statistics import mean
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import EvalCaseResult, PromptRewrite
from app.prompts import DIMENSION_PROMPT_MAP, get_prompt


class MetaAgent:
    agent_id = "meta"
    prompt_name = "meta"

    def propose_rewrite(self, session: Session, eval_run_id: str) -> PromptRewrite | None:
        cases = session.scalars(select(EvalCaseResult).where(EvalCaseResult.eval_run_id == eval_run_id)).all()
        failed = [case for case in cases if not case.passed]
        if not failed:
            return None
        dimension_scores: dict[str, list[float]] = {}
        dimension_reasons: dict[str, list[str]] = {}
        for case in failed:
            for dimension, payload in case.scores.items():
                dimension_scores.setdefault(dimension, []).append(float(payload.get("score", 0.0)))
                dimension_reasons.setdefault(dimension, []).append(str(payload.get("justification", "")))
        worst_dimension = min(dimension_scores, key=lambda name: mean(dimension_scores[name]))
        prompt_name = DIMENSION_PROMPT_MAP.get(worst_dimension, "synthesis")
        prompt = get_prompt(session, prompt_name)
        appended = self._rewrite_instruction(worst_dimension, dimension_reasons.get(worst_dimension, []))
        proposed_text = f"{prompt.text}\n\nAdditional guardrail from eval feedback: {appended}"
        rewrite = PromptRewrite(
            eval_run_id=eval_run_id,
            prompt_name=prompt_name,
            base_version=prompt.version,
            proposed_text=proposed_text,
            structured_diff={
                "operation": "append_guardrail",
                "target_dimension": worst_dimension,
                "before_version": prompt.version,
                "added_text": appended,
                "failed_case_count": len(failed),
            },
            justification=(
                f"The lowest average failed-case score was {worst_dimension} "
                f"({mean(dimension_scores[worst_dimension]):.2f}). The rewrite makes that criterion explicit."
            ),
            status="pending",
        )
        session.add(rewrite)
        session.flush()
        return rewrite

    def _rewrite_instruction(self, dimension: str, reasons: list[str]) -> str:
        if dimension == "citation_accuracy":
            return "Before emitting an answer claim, verify that at least two cited chunks support it and name each chunk's contribution."
        if dimension == "contradiction_resolution":
            return "When critique flags a span, either remove it or replace it with the corrected premise; never surface unresolved disagreement."
        if dimension == "tool_selection_efficiency":
            return "Select the smallest sufficient tool set for each subtask and avoid repeated calls once adequate evidence exists."
        if dimension == "context_budget_compliance":
            return "Check remaining context budget before adding verbose context and trigger compression before overflow."
        if dimension == "critique_agreement":
            return "Use critique feedback as a blocking gate for final claims unless the synthesis can cite stronger evidence."
        return "Prefer directly verifiable claims and mark unsupported premises before answering."

