from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas import ToolResult, ToolStatus


class StructuredDataLookupTool:
    name = "structured_data_lookup"

    def run(self, session: Session, payload: dict[str, Any]) -> ToolResult:
        question = str(payload.get("question", "")).strip().lower()
        sql = str(payload.get("sql", "")).strip()
        if payload.get("simulate_timeout"):
            return ToolResult(status=ToolStatus.timeout, error_code="lookup_timeout", message="Local database lookup timed out")
        if sql:
            generated_sql = sql
        else:
            generated_sql = self._nl_to_sql(question)
        if not generated_sql:
            return ToolResult(
                status=ToolStatus.malformed,
                error_code="lookup_malformed_question",
                message="Could not convert natural language input to a safe SQL query",
            )
        if not self._safe_select(generated_sql):
            return ToolResult(status=ToolStatus.malformed, error_code="lookup_unsafe_sql", message="Only safe SELECT lookups are allowed")

        rows = session.execute(text(generated_sql)).mappings().all()
        result_rows = [dict(row) for row in rows]
        if not result_rows:
            return ToolResult(
                status=ToolStatus.empty,
                error_code="lookup_empty",
                message="Safe SQL query returned no rows",
                payload={"sql": generated_sql, "rows": []},
            )
        return ToolResult(status=ToolStatus.ok, payload={"sql": generated_sql, "rows": result_rows})

    def _nl_to_sql(self, question: str) -> str | None:
        cases = [
            (("capital", "france"), "subject = 'france' AND predicate = 'capital'"),
            (("hamlet", "wrote"), "subject = 'hamlet' AND predicate = 'author'"),
            (("hamlet", "author"), "subject = 'hamlet' AND predicate = 'author'"),
            (("water", "boil"), "subject = 'water' AND predicate = 'boiling_point_celsius_at_1_atm'"),
            (("speed", "light"), "subject = 'light' AND predicate = 'speed_m_per_s'"),
            (("largest", "planet"), "subject = 'solar system' AND predicate = 'largest_planet'"),
            (("postgres", "acid"), "subject = 'postgresql' AND predicate = 'supports'"),
            (("sqlite", "embedded"), "subject = 'sqlite' AND predicate = 'deployment_type'"),
        ]
        for terms, where_clause in cases:
            if all(term in question for term in terms):
                return (
                    "SELECT subject, predicate, value, source_url "
                    f"FROM structured_facts WHERE {where_clause} ORDER BY id LIMIT 5"
                )
        normalized = re.sub(r"[^a-z0-9 ]+", " ", question)
        words = [word for word in normalized.split() if len(word) > 3]
        if not words:
            return None
        like = " OR ".join(
            f"lower(subject || ' ' || predicate || ' ' || value) LIKE '%{word}%'" for word in words[:4]
        )
        return (
            "SELECT subject, predicate, value, source_url "
            f"FROM structured_facts WHERE {like} ORDER BY id LIMIT 5"
        )

    def _safe_select(self, sql: str) -> bool:
        lowered = sql.lower().strip()
        if not lowered.startswith("select "):
            return False
        forbidden = (";", "insert", "update", "delete", "drop", "alter", "pragma", "attach")
        if any(token in lowered for token in forbidden):
            return False
        return "from structured_facts" in lowered

