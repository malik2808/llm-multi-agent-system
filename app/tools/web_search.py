from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import KnowledgeChunk
from app.schemas import ToolResult, ToolStatus


WORD_RE = re.compile(r"[a-z0-9]+")


class WebSearchStub:
    name = "web_search_stub"

    def run(self, session: Session, payload: dict[str, Any]) -> ToolResult:
        query = str(payload.get("query", "")).strip()
        if payload.get("simulate_timeout"):
            return ToolResult(status=ToolStatus.timeout, error_code="web_timeout", message="Search stub timed out")
        if not query:
            return ToolResult(status=ToolStatus.malformed, error_code="web_malformed_query", message="query is required")

        query_terms = set(WORD_RE.findall(query.lower()))
        if not query_terms:
            return ToolResult(status=ToolStatus.malformed, error_code="web_malformed_query", message="query had no searchable terms")

        chunks = session.scalars(select(KnowledgeChunk)).all()
        scored: list[dict[str, Any]] = []
        for chunk in chunks:
            haystack = f"{chunk.title} {chunk.text} {' '.join(chunk.tags)}".lower()
            terms = set(WORD_RE.findall(haystack))
            overlap = query_terms & terms
            if overlap:
                score = len(overlap) / max(1, len(query_terms))
                scored.append(
                    {
                        "chunk_id": chunk.id,
                        "title": chunk.title,
                        "url": chunk.url,
                        "snippet": chunk.text,
                        "relevance_score": round(score, 3),
                        "matched_terms": sorted(overlap),
                    }
                )
        scored.sort(key=lambda item: item["relevance_score"], reverse=True)
        limit = int(payload.get("limit", 5))
        results = scored[:limit]
        if not results:
            return ToolResult(status=ToolStatus.empty, error_code="web_empty", message="No structured search results matched")
        return ToolResult(status=ToolStatus.ok, payload={"results": results})

