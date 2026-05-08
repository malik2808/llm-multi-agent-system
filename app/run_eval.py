from __future__ import annotations

import json

from app.db import init_db, session_scope
from app.eval_harness import EvalHarness
from app.seed import seed_reference_data


def main() -> None:
    init_db()
    with session_scope() as session:
        seed_reference_data(session)
        run = EvalHarness().run(session)
        print(json.dumps({"eval_run_id": run.id, "summary": run.summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

