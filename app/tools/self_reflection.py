from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from app.schemas import ToolResult, ToolStatus


SUBJECT_RE = re.compile(r"\b(france|paris|hamlet|water|light|jupiter|postgresql|sqlite)\b", re.I)


class SelfReflectionTool:
    name = "self_reflection"

    def run(self, session: Session, payload: dict[str, Any]) -> ToolResult:
        del session
        outputs = payload.get("agent_outputs")
        if not isinstance(outputs, dict):
            return ToolResult(status=ToolStatus.malformed, error_code="reflection_malformed", message="agent_outputs dict is required")
        claims: list[dict[str, Any]] = []
        for agent_id, output in outputs.items():
            for claim in output.get("claims", []):
                text = str(claim.get("text", ""))
                subject_match = SUBJECT_RE.search(text)
                claims.append(
                    {
                        "agent_id": agent_id,
                        "text": text,
                        "subject": subject_match.group(1).lower() if subject_match else None,
                    }
                )

        contradictions = []
        for left in claims:
            for right in claims:
                if left is right or not left["subject"] or left["subject"] != right["subject"]:
                    continue
                left_lower = left["text"].lower()
                right_lower = right["text"].lower()
                if ("not " in left_lower and "not " not in right_lower) or ("not " in right_lower and "not " not in left_lower):
                    contradictions.append(
                        {
                            "left_agent": left["agent_id"],
                            "right_agent": right["agent_id"],
                            "left_span": left["text"],
                            "right_span": right["text"],
                            "reason": "same subject has negated and non-negated claims",
                        }
                    )
        if not contradictions:
            return ToolResult(status=ToolStatus.empty, error_code="reflection_no_contradictions", message="No contradictions found")
        return ToolResult(status=ToolStatus.ok, payload={"contradictions": contradictions})

