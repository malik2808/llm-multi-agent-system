from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.seed import seed_reference_data
from app.schemas import ToolStatus
from app.tools import CodeExecutionSandbox, StructuredDataLookupTool, WebSearchStub


def make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    seed_reference_data(session)
    session.commit()
    return session


def test_web_search_stub_returns_structured_results() -> None:
    session = make_session()
    result = WebSearchStub().run(session, {"query": "France capital", "limit": 2})

    assert result.status == ToolStatus.ok
    assert result.payload["results"][0]["url"].startswith("https://example.test/")
    assert "relevance_score" in result.payload["results"][0]


def test_structured_lookup_contracts() -> None:
    session = make_session()
    ok = StructuredDataLookupTool().run(session, {"question": "What is the capital of France?"})
    malformed = StructuredDataLookupTool().run(session, {"question": ""})

    assert ok.status == ToolStatus.ok
    assert ok.payload["rows"][0]["value"] == "Paris"
    assert malformed.status == ToolStatus.malformed


def test_code_sandbox_reports_stdout_and_exit_code() -> None:
    result = CodeExecutionSandbox().run(make_session(), {"code": "print(3 + 5)"})

    assert result.status == ToolStatus.ok
    assert result.payload["stdout"].strip() == "8"
    assert result.payload["exit_code"] == 0

