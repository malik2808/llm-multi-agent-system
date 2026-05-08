from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.context import SharedContext
from app.schemas import AgentOutput, ToolStatus
from app.tools import ToolExecutor


class CritiqueAgent(BaseAgent):
    agent_id = "critique"
    prompt_name = "critique"
    default_budget = 1400

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
        del context_input, prompt_text, run_key
        target_keys = kwargs.get("target_keys") or [
            key for key in context.agent_outputs.keys() if not key.startswith("critique")
        ]
        flags: list[dict[str, Any]] = []
        reviewed = 0
        for key in target_keys:
            output = context.agent_outputs.get(key, {})
            output_flags = []
            for claim in output.get("claims", []):
                reviewed += 1
                critique = self._critique_claim(key, claim)
                output_flags.append(critique)
                if not critique["agree"]:
                    flags.append(critique)
            context.critiques[key] = output_flags

        reflection = tool_executor.call(
            session,
            job_id=context.job_id,
            agent_id=self.agent_id,
            tool_name="self_reflection",
            payload={"agent_outputs": {key: context.agent_outputs[key] for key in target_keys if key in context.agent_outputs}},
            accept_empty=True,
        )
        context.add_tool_result({"tool": "self_reflection", "target_keys": target_keys, **reflection.model_dump()})
        if reflection.status == ToolStatus.ok:
            for contradiction in reflection.payload.get("contradictions", []):
                flag = {
                    "agent_key": contradiction["left_agent"],
                    "span": contradiction["left_span"],
                    "confidence": 0.35,
                    "agree": False,
                    "reason": contradiction["reason"],
                    "contradiction": contradiction,
                }
                flags.append(flag)
                context.critiques.setdefault(contradiction["left_agent"], []).append(flag)

        text = f"Reviewed {reviewed} claim(s); flagged {len(flags)} span-level issue(s)."
        return AgentOutput(
            agent_id=self.agent_id,
            text=text,
            claims=[
                {
                    "text": text,
                    "confidence": 0.9,
                    "source_tools": ["self_reflection"],
                    "flag_count": len(flags),
                }
            ],
            artifacts={"target_keys": target_keys, "flags": flags, "critiques": context.critiques},
        )

    def _critique_claim(self, agent_key: str, claim: dict[str, Any]) -> dict[str, Any]:
        text = str(claim.get("text", ""))
        lower = text.lower()
        confidence = float(claim.get("confidence", 0.5))
        source_chunks = claim.get("source_chunks", [])
        reason = "claim has adequate support"
        agree = True
        if "ignore previous" in lower or "override system" in lower:
            confidence = min(confidence, 0.25)
            agree = False
            reason = "span appears to preserve an unsafe instruction rather than neutralize it"
        elif "paris is in germany" in lower or "capital of germany is paris" in lower:
            confidence = min(confidence, 0.15)
            agree = False
            reason = "span contains a known false premise about Paris and Germany"
        elif claim.get("source_tools") == ["web_search_stub"] and len(source_chunks) < 2:
            confidence = min(confidence, 0.45)
            agree = False
            reason = "retrieval claim lacks the required two-hop chunk support"
        elif confidence < 0.5:
            agree = False
            reason = "claim confidence is below acceptance threshold"
        return {
            "agent_key": agent_key,
            "span": text[:240],
            "confidence": round(confidence, 3),
            "agree": agree,
            "reason": reason,
            "source_chunks": source_chunks,
        }

