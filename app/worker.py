from __future__ import annotations

import time

from sqlalchemy import select

from app.config import get_settings
from app.db import init_db, session_scope
from app.eval_harness import EvalHarness
from app.models import EvalRun, Job
from app.orchestrator import Orchestrator
from app.seed import seed_reference_data


def run_once(orchestrator: Orchestrator) -> bool:
    with session_scope() as session:
        job = session.scalars(select(Job).where(Job.status == "queued").order_by(Job.created_at.asc()).limit(1)).first()
        if not job:
            return False
        orchestrator.run_job(session, job)
        return True


def main() -> None:
    settings = get_settings()
    init_db()
    with session_scope() as session:
        seed_reference_data(session)
        existing_eval = session.scalars(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(1)).first()
        if settings.run_eval_on_start and not existing_eval:
            EvalHarness().run(session)

    orchestrator = Orchestrator()
    while True:
        did_work = run_once(orchestrator)
        if not did_work:
            time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()

