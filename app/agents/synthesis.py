from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.context import SharedContext
from app.schemas import AgentOutput
from app.tools import ToolExecutor
from app.utils import sentence_split


class SynthesisAgent(BaseAgent):
    agent_id = "synthesis"
    prompt_name = "synthesis"
    default_budget = 1500

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
        del session, context_input, prompt_text, tool_executor
        revision = bool(kwargs.get("revision"))
        accepted_claims = self._accepted_claims(context)
        if accepted_claims:
            sentences = [claim["text"].strip() for claim in accepted_claims if claim.get("text")]
        else:
            sentences = ["I do not have enough grounded evidence to answer confidently."]
        if len(sentences) > 3:
            sentences = sentences[:3]
        final_text = " ".join(sentence if sentence.endswith((".", "!", "?")) else f"{sentence}." for sentence in sentences)
        provenance = []
        for sentence in sentence_split(final_text):
            source_claim = next((claim for claim in accepted_claims if sentence.rstrip(".") in claim.get("text", "")), accepted_claims[0] if accepted_claims else {})
            provenance.append(
                {
                    "sentence": sentence,
                    "source_agent": source_claim.get("source_agent", "synthesis"),
                    "source_task_id": source_claim.get("task_id"),
                    "source_chunks": source_claim.get("source_chunks", []),
                    "source_tools": source_claim.get("source_tools", []),
                }
            )
        context.final_answer = final_text
        context.provenance_map = provenance
        text = final_text if not revision else f"Revision after critique: {final_text}"
        return AgentOutput(
            agent_id=self.agent_id,
            text=text,
            claims=[
                {
                    "text": sentence,
                    "confidence": self._sentence_confidence(sentence, accepted_claims),
                    "source_agent": item["source_agent"],
                    "source_chunks": item["source_chunks"],
                    "source_tools": item["source_tools"],
                }
                for sentence, item in zip(sentence_split(final_text), provenance)
            ],
            artifacts={"final_answer": final_text, "provenance_map": provenance, "revision": revision},
        )

    def _accepted_claims(self, context: SharedContext) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []
        disagreed_spans = {
            critique["span"]
            for critiques in context.critiques.values()
            for critique in critiques
            if not critique.get("agree", True)
        }
        for agent_key, output in context.agent_outputs.items():
            if not agent_key.startswith("retrieval"):
                continue
            for claim in output.get("claims", []):
                text = str(claim.get("text", ""))
                if any(text[:120] in span or span[:120] in text for span in disagreed_spans):
                    continue
                enriched = dict(claim)
                enriched["source_agent"] = agent_key
                accepted.append(enriched)
        accepted.sort(key=lambda claim: float(claim.get("confidence", 0.0)), reverse=True)
        return accepted

    def _sentence_confidence(self, sentence: str, accepted_claims: list[dict[str, Any]]) -> float:
        for claim in accepted_claims:
            if sentence.rstrip(".") in claim.get("text", ""):
                return float(claim.get("confidence", 0.7))
        return 0.65

