from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.eval_harness import EvalHarness
from app.models import Base
from app.seed import seed_reference_data


def test_eval_harness_scores_one_case() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    session = Session()
    seed_reference_data(session)

    run = EvalHarness().run(session, case_ids=["baseline_1"], targeted=True)

    assert run.summary["case_count"] == 1
    assert "baseline_1" not in run.summary["failed_cases"]

