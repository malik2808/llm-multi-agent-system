from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.utils import estimate_tokens


@dataclass
class ContextItem:
    kind: str
    content: Any
    agent_id: str | None = None
    structured: bool = False


@dataclass
class SharedContext:
    job_id: str
    query: str
    items: list[ContextItem] = field(default_factory=list)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    critiques: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    routing_decisions: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    provenance_map: list[dict[str, Any]] = field(default_factory=list)
    compressed_summary: str = ""

    def add_item(self, kind: str, content: Any, *, agent_id: str | None = None, structured: bool = False) -> None:
        self.items.append(ContextItem(kind=kind, content=content, agent_id=agent_id, structured=structured))

    def add_agent_output(self, agent_id: str, output: dict[str, Any]) -> None:
        self.agent_outputs[agent_id] = output
        self.add_item("agent_output", output, agent_id=agent_id, structured=True)

    def add_tool_result(self, result: dict[str, Any]) -> None:
        self.tool_results.append(result)
        self.add_item("tool_result", result, structured=True)

    def export(self) -> dict[str, Any]:
        recent_filler = [
            {
                "kind": item.kind,
                "agent_id": item.agent_id,
                "content": str(item.content)[:400],
            }
            for item in self.items
            if not item.structured
        ][-6:]
        return {
            "job_id": self.job_id,
            "query": self.query,
            "compressed_summary": self.compressed_summary,
            "tasks": self.tasks,
            "agent_outputs": self.agent_outputs,
            "critiques": self.critiques,
            "tool_results": self.tool_results,
            "routing_decisions": self.routing_decisions,
            "final_answer": self.final_answer,
            "provenance_map": self.provenance_map,
            "recent_filler": recent_filler,
        }

    def token_count(self) -> int:
        return estimate_tokens(self.export())


class ContextBudgetManager:
    def __init__(self) -> None:
        self.declared_budgets: dict[str, int] = {}
        self.used_tokens: dict[str, int] = {}
        self.policy_violations: list[dict[str, Any]] = []

    def declare_budget(self, agent_id: str, max_tokens: int) -> None:
        self.declared_budgets[agent_id] = max_tokens
        self.used_tokens.setdefault(agent_id, 0)

    def remaining(self, agent_id: str) -> int:
        budget = self.declared_budgets.get(agent_id, 0)
        return max(0, budget - self.used_tokens.get(agent_id, 0))

    def check_remaining(self, agent_id: str, proposed_content: Any) -> int:
        return self.remaining(agent_id) - estimate_tokens(proposed_content)

    def add_usage(self, agent_id: str, content: Any) -> list[dict[str, Any]]:
        added = estimate_tokens(content)
        self.used_tokens[agent_id] = self.used_tokens.get(agent_id, 0) + added
        budget = self.declared_budgets.get(agent_id, 0)
        if budget and self.used_tokens[agent_id] > budget:
            violation = {
                "type": "context_budget_overflow",
                "agent_id": agent_id,
                "budget": budget,
                "used": self.used_tokens[agent_id],
                "overflow": self.used_tokens[agent_id] - budget,
            }
            self.policy_violations.append(violation)
            return [violation]
        return []

    def assemble_for_agent(self, context: SharedContext, agent_id: str) -> tuple[dict[str, Any], int]:
        payload = context.export()
        return payload, estimate_tokens(payload)

    def enforce_before_run(self, context: SharedContext, agent_id: str, compression_agent: Any) -> dict[str, Any]:
        budget = self.declared_budgets.get(agent_id, 0)
        payload, token_count = self.assemble_for_agent(context, agent_id)
        if budget and token_count > budget:
            compression_agent.compress(context, target_tokens=max(64, budget // 2))
            payload, token_count = self.assemble_for_agent(context, agent_id)
        if budget and token_count > budget:
            violation = {
                "type": "context_assembly_over_budget",
                "agent_id": agent_id,
                "budget": budget,
                "assembled_tokens": token_count,
            }
            self.policy_violations.append(violation)
        self.used_tokens[agent_id] = token_count
        return payload
