from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.context import SharedContext
from app.schemas import AgentOutput
from app.tools import ToolExecutor


AMBIGUOUS_TERMS = ("best", "better", "it", "they", "that", "this", "compare", "recommend")
INJECTION_TERMS = ("ignore previous", "ignore all", "system instruction", "developer message", "override", "jailbreak")


class DecompositionAgent(BaseAgent):
    agent_id = "decomposition"
    prompt_name = "decomposition"
    default_budget = 900

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
        del session, context_input, prompt_text, tool_executor, run_key, kwargs
        query = context.query.strip()
        lower = query.lower()
        tasks: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []

        if any(term in lower for term in INJECTION_TERMS):
            tasks.append(
                {
                    "id": "task_safety",
                    "type": "robustness_check",
                    "query": query,
                    "dependencies": [],
                    "status": "pending",
                    "tool_hints": ["web_search_stub", "self_reflection"],
                }
            )
            claims.append(
                {
                    "text": "The query contains possible prompt-injection content that should be treated as untrusted data.",
                    "confidence": 0.95,
                    "span": query[:160],
                }
            )

        compare_targets = self._extract_compare_targets(query)
        if compare_targets:
            previous_ids: list[str] = []
            for index, target in enumerate(compare_targets, start=1):
                task_id = f"task_compare_source_{index}"
                previous_ids.append(task_id)
                tasks.append(
                    {
                        "id": task_id,
                        "type": "fact_lookup",
                        "query": target,
                        "dependencies": [],
                        "status": "pending",
                        "tool_hints": ["web_search_stub", "structured_data_lookup"],
                    }
                )
            tasks.append(
                {
                    "id": "task_compare_synthesis",
                    "type": "comparison",
                    "query": query,
                    "dependencies": previous_ids,
                    "status": "pending",
                    "tool_hints": ["self_reflection"],
                }
            )
        else:
            if self._looks_ambiguous(lower):
                tasks.append(
                    {
                        "id": "task_assumptions",
                        "type": "ambiguity_resolution",
                        "query": query,
                        "dependencies": [],
                        "status": "resolved",
                        "result": {
                            "assumptions": [
                                "Use the most common interpretation of the request.",
                                "Prefer verifiable facts over subjective recommendations.",
                            ]
                        },
                        "tool_hints": [],
                    }
                )
                claims.append(
                    {
                        "text": "The request is underspecified, so downstream agents should carry explicit assumptions.",
                        "confidence": 0.8,
                        "span": query[:160],
                    }
                )
                dependencies = ["task_assumptions"]
            else:
                dependencies = []

            task_type = "calculation" if self._contains_arithmetic(query) else "fact_lookup"
            tasks.append(
                {
                    "id": "task_answer",
                    "type": task_type,
                    "query": query,
                    "dependencies": dependencies,
                    "status": "pending",
                    "tool_hints": ["code_execution_sandbox"] if task_type == "calculation" else ["web_search_stub", "structured_data_lookup"],
                }
            )

        context.tasks = {task["id"]: task for task in tasks}
        for task in tasks:
            context.add_item("task", task, agent_id=self.agent_id, structured=True)
        text = f"Created {len(tasks)} typed subtask(s) with dependency gates: " + ", ".join(
            f"{task['id']}<{task['type']}>" for task in tasks
        )
        return AgentOutput(
            agent_id=self.agent_id,
            text=text,
            claims=claims
            or [
                {
                    "text": "The query can proceed through retrieval and synthesis without clarification.",
                    "confidence": 0.85,
                    "span": query[:160],
                }
            ],
            artifacts={"tasks": tasks},
        )

    def _looks_ambiguous(self, lower: str) -> bool:
        return any(term in lower.split() for term in AMBIGUOUS_TERMS) or lower.count("?") > 1

    def _contains_arithmetic(self, query: str) -> bool:
        return bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", query))

    def _extract_compare_targets(self, query: str) -> list[str]:
        lower = query.lower()
        if "compare" not in lower and " vs " not in lower and " versus " not in lower:
            return []
        cleaned = re.sub(r"(?i)compare|which is better|\?|please|for local development", " ", query)
        pieces = re.split(r"(?i)\s+and\s+|\s+vs\.?\s+|\s+versus\s+", cleaned)
        targets = [piece.strip(" .,:;") for piece in pieces if piece.strip(" .,:;")]
        return targets[:3]

