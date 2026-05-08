from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import KnowledgeChunk, StructuredFact
from app.prompts import ensure_default_prompts


CHUNKS = [
    {
        "id": "chunk_france_capital",
        "title": "France overview",
        "url": "https://example.test/france",
        "text": "France is a country in Western Europe. Its capital city is Paris.",
        "tags": ["france", "capital", "europe"],
    },
    {
        "id": "chunk_paris_government",
        "title": "Paris civic role",
        "url": "https://example.test/paris",
        "text": "Paris hosts the national government of France and is commonly cited as France's capital.",
        "tags": ["paris", "france", "government"],
    },
    {
        "id": "chunk_hamlet_author",
        "title": "Hamlet authorship",
        "url": "https://example.test/hamlet",
        "text": "Hamlet is a tragedy written by William Shakespeare.",
        "tags": ["hamlet", "shakespeare", "literature"],
    },
    {
        "id": "chunk_shakespeare_context",
        "title": "Shakespeare works",
        "url": "https://example.test/shakespeare",
        "text": "William Shakespeare wrote plays including Hamlet, Macbeth, and Othello.",
        "tags": ["shakespeare", "plays", "hamlet"],
    },
    {
        "id": "chunk_water_boiling",
        "title": "Water phase change",
        "url": "https://example.test/water",
        "text": "At standard atmospheric pressure, pure water boils at 100 degrees Celsius.",
        "tags": ["water", "boiling", "science"],
    },
    {
        "id": "chunk_standard_pressure",
        "title": "Standard pressure note",
        "url": "https://example.test/pressure",
        "text": "Boiling points depend on pressure, so the common 100 C value assumes one atmosphere.",
        "tags": ["pressure", "boiling", "science"],
    },
    {
        "id": "chunk_speed_light",
        "title": "Speed of light",
        "url": "https://example.test/light",
        "text": "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        "tags": ["physics", "light", "constant"],
    },
    {
        "id": "chunk_si_definition",
        "title": "SI definition",
        "url": "https://example.test/si",
        "text": "The meter is defined using the fixed numerical value of the speed of light in vacuum.",
        "tags": ["physics", "si", "light"],
    },
    {
        "id": "chunk_jupiter",
        "title": "Solar system planets",
        "url": "https://example.test/jupiter",
        "text": "Jupiter is the largest planet in the Solar System by mass and volume.",
        "tags": ["jupiter", "planet", "solar system"],
    },
    {
        "id": "chunk_planet_sizes",
        "title": "Planet size comparison",
        "url": "https://example.test/planets",
        "text": "Among the eight planets, Jupiter exceeds the others in diameter and mass.",
        "tags": ["planet", "jupiter", "comparison"],
    },
    {
        "id": "chunk_postgres_acid",
        "title": "PostgreSQL reliability",
        "url": "https://example.test/postgres",
        "text": "PostgreSQL supports ACID transactions and relational SQL queries.",
        "tags": ["postgres", "database", "acid"],
    },
    {
        "id": "chunk_sqlite_local",
        "title": "SQLite local storage",
        "url": "https://example.test/sqlite",
        "text": "SQLite is an embedded SQL database often used for local development and tests.",
        "tags": ["sqlite", "database", "local"],
    },
    {
        "id": "chunk_prompt_injection",
        "title": "Prompt injection defense",
        "url": "https://example.test/prompt-injection",
        "text": "Prompt injection attempts should be treated as untrusted user content, not system instructions.",
        "tags": ["security", "prompt injection", "llm"],
    },
    {
        "id": "chunk_wrong_premise",
        "title": "Wrong premise handling",
        "url": "https://example.test/wrong-premise",
        "text": "When a user question contains a false premise, a robust assistant should correct the premise before answering.",
        "tags": ["reasoning", "premise", "robustness"],
    },
    {
        "id": "chunk_context_budget",
        "title": "Context budget management",
        "url": "https://example.test/context-budget",
        "text": "A context budget manager should track token use and summarize older low-value context before overflow.",
        "tags": ["context", "tokens", "llm"],
    },
]

FACTS = [
    ("france", "capital", "Paris", "https://example.test/france"),
    ("hamlet", "author", "William Shakespeare", "https://example.test/hamlet"),
    ("water", "boiling_point_celsius_at_1_atm", "100", "https://example.test/water"),
    ("light", "speed_m_per_s", "299792458", "https://example.test/light"),
    ("solar system", "largest_planet", "Jupiter", "https://example.test/jupiter"),
    ("postgresql", "supports", "ACID transactions", "https://example.test/postgres"),
    ("sqlite", "deployment_type", "embedded database", "https://example.test/sqlite"),
]


def seed_reference_data(session: Session) -> None:
    ensure_default_prompts(session)
    for item in CHUNKS:
        existing = session.get(KnowledgeChunk, item["id"])
        if existing:
            existing.title = item["title"]
            existing.url = item["url"]
            existing.text = item["text"]
            existing.tags = item["tags"]
        else:
            session.add(KnowledgeChunk(**item))

    for subject, predicate, value, source_url in FACTS:
        existing = session.scalar(
            select(StructuredFact).where(
                StructuredFact.subject == subject,
                StructuredFact.predicate == predicate,
            )
        )
        if existing:
            existing.value = value
            existing.source_url = source_url
        else:
            session.add(
                StructuredFact(
                    subject=subject,
                    predicate=predicate,
                    value=value,
                    source_url=source_url,
                )
            )
    session.flush()

