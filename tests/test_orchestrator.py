from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Job, ToolCallLog
from app.orchestrator import Orchestrator
from app.seed import seed_reference_data


def test_orchestrator_answers_with_provenance_and_tool_logs() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    session = Session()
    seed_reference_data(session)
    job = Job(query="What is the capital of France?", status="queued")
    session.add(job)
    session.commit()

    context = Orchestrator().run_job(session, job)

    assert job.status == "completed"
    assert "Paris" in (job.final_answer or "")
    assert context.provenance_map
    assert session.query(ToolCallLog).filter_by(job_id=job.id).count() >= 2

