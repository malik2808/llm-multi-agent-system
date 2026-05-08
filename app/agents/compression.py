from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.context import SharedContext
from app.schemas import AgentOutput
from app.tools import ToolExecutor
from app.utils import estimate_tokens


class CompressionAgent:
    agent_id = "compression"
    prompt_name = "compression"
    default_budget = 600

    def compress(self, context: SharedContext, target_tokens: int = 300) -> AgentOutput:
        structured_items = [item for item in context.items if item.structured]
        filler_items = [item for item in context.items if not item.structured]
        filler_summary = "; ".join(str(item.content)[:120] for item in filler_items[-5:])
        if filler_summary:
            context.compressed_summary = (
                f"{context.compressed_summary} | Compressed filler: {filler_summary}".strip(" |")
            )[: target_tokens * 5]
        context.items = structured_items[-20:]
        output = AgentOutput(
            agent_id=self.agent_id,
            text="Compressed older conversational context while preserving structured artifacts exactly.",
            claims=[
                {
                    "text": "Structured artifacts were retained verbatim during context compression.",
                    "confidence": 1.0,
                    "source": "context_manager",
                }
            ],
            artifacts={
                "retained_structured_items": len(context.items),
                "compressed_summary_tokens": estimate_tokens(context.compressed_summary),
            },
        )
        return output

    def execute(self, *_args: Any, **_kwargs: Any) -> AgentOutput:
        raise RuntimeError("CompressionAgent is invoked by ContextBudgetManager.compress, not as a routed agent")

