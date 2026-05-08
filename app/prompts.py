from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import PromptVersion


DEFAULT_PROMPTS: dict[str, str] = {
    "decomposition": (
        "Break the user's request into typed subtasks. Include dependency edges, "
        "ambiguity notes, and which tools may be useful. Do not answer the user."
    ),
    "retrieval": (
        "Answer assigned subtasks using at least two retrieved chunks when retrieval "
        "is needed. Cite which chunk supports each claim and reject unsupported facts."
    ),
    "critique": (
        "Review agent outputs claim by claim. Assign confidence per span, flag exact "
        "text spans you dispute, and explain the evidence standard used."
    ),
    "synthesis": (
        "Merge agent outputs into a concise final answer. Resolve contradictions and "
        "attach a provenance map from each sentence to source agents and chunks."
    ),
    "compression": (
        "Compress conversational filler while preserving structured data exactly: "
        "tool outputs, scores, citations, claims, and policy violations."
    ),
    "meta": (
        "Inspect evaluation failures, find the worst prompt by scoring dimension, "
        "and propose a concrete rewrite with a structured diff. Do not apply it."
    ),
}


DIMENSION_PROMPT_MAP = {
    "answer_correctness": "retrieval",
    "citation_accuracy": "retrieval",
    "contradiction_resolution": "synthesis",
    "tool_selection_efficiency": "decomposition",
    "context_budget_compliance": "compression",
    "critique_agreement": "critique",
}


def ensure_default_prompts(session: Session) -> None:
    for prompt_name, text in DEFAULT_PROMPTS.items():
        exists = session.scalar(
            select(PromptVersion).where(
                PromptVersion.prompt_name == prompt_name,
                PromptVersion.version == 1,
            )
        )
        if not exists:
            session.add(
                PromptVersion(
                    prompt_name=prompt_name,
                    version=1,
                    text=text,
                    status="approved",
                )
            )
    session.flush()


def get_prompt(session: Session, prompt_name: str) -> PromptVersion:
    prompt = session.scalars(
        select(PromptVersion)
        .where(PromptVersion.prompt_name == prompt_name, PromptVersion.status == "approved")
        .order_by(PromptVersion.version.desc())
        .limit(1)
    ).first()
    if prompt is None:
        ensure_default_prompts(session)
        prompt = session.scalars(
            select(PromptVersion)
            .where(PromptVersion.prompt_name == prompt_name, PromptVersion.status == "approved")
            .order_by(PromptVersion.version.desc())
            .limit(1)
        ).first()
    if prompt is None:
        raise ValueError(f"Unknown prompt: {prompt_name}")
    return prompt


def approved_prompt_versions(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(PromptVersion.prompt_name, func.max(PromptVersion.version))
        .where(PromptVersion.status == "approved")
        .group_by(PromptVersion.prompt_name)
    ).all()
    return {name: version for name, version in rows}


def approve_prompt(session: Session, prompt_name: str, proposed_text: str) -> PromptVersion:
    latest = session.scalar(
        select(func.max(PromptVersion.version)).where(PromptVersion.prompt_name == prompt_name)
    )
    version = int(latest or 0) + 1
    prompt = PromptVersion(
        prompt_name=prompt_name,
        version=version,
        text=proposed_text,
        status="approved",
    )
    session.add(prompt)
    session.flush()
    return prompt

