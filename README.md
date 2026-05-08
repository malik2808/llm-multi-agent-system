# LLM Engineer Take-Home Assessment

This repository implements a containerized multi-agent system with dynamic routing, tool failure contracts, context budget enforcement, an evaluation harness, prompt rewrite approvals, SSE streaming, and queryable execution traces.

The implementation is intentionally deterministic: no external LLM key is required, so a reviewer can run it in a few minutes and diff eval output across runs. The agent interfaces, context discipline, logging, and prompt-improvement workflow mirror a production LLM system while keeping the model behavior inspectable.

## Quick Start

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Log query UI: `http://localhost:8080` using Adminer. Server is `db`, database/user default to the env values in `.env.example`.
- Worker: polls queued jobs and runs the eval harness once on first startup when `RUN_EVAL_ON_START=true`.
- Database: PostgreSQL 16.

No credentials are hardcoded. The development compose file uses PostgreSQL trust auth by default. For production-like runs, set `DATABASE_URL`, `POSTGRES_USER`, `POSTGRES_DB`, and a secure PostgreSQL auth method through environment variables.

Local development:

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e ".[dev]"
python -m app.run_eval
uvicorn app.main:app --reload
python -m app.worker
```

## API

Exactly five application endpoints are exposed. FastAPI's built-in docs route is disabled to keep the API surface explicit.

### `POST /query`

Submits a query and returns a Server-Sent Events stream. The stream includes `routing_decision`, `agent_started`, `stream_token`, `tool_call`, `agent_completed`, and terminal job events.

Request:

```json
{"query": "What is the capital of France?"}
```

### `GET /trace/{job_id}`

Returns the ordered execution trace for a job: structured events, tool calls, exact prompts sent to agents, exact outputs, token counts, hashes, latencies, and policy violations.

### `GET /eval/latest`

Returns the most recent eval summary broken down by category and dimension.

### `POST /prompt-rewrites/{rewrite_id}/decision`

Approves or rejects a pending meta-agent prompt rewrite.

```json
{"decision": "approve", "reason": "Improves citation gating."}
```

### `POST /eval/rerun-failed`

Runs a targeted eval on the latest failed cases using the latest approved prompts and stores performance deltas against the source eval run.

All error responses use:

```json
{"error_code": "machine_readable_code", "message": "Human readable message.", "job_id": "job_id_if_applicable"}
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the text diagram.

The orchestrator is the only component that invokes agents. Agents never call each other. All handoffs go through `SharedContext`, which carries the task graph, tool results, agent outputs, critique spans, routing decisions, final answer, and sentence-level provenance.

## Agents

- Decomposition agent: turns ambiguous input into typed subtasks and dependency edges. It marks assumption tasks resolved and leaves dependent tasks pending until prerequisites complete.
- Retrieval agent: uses the tool executor for search, structured lookup, code execution, and explicit fallback. Retrieval answers cite at least two chunks when chunk retrieval is used.
- Critique agent: reviews each prior output at claim/span level, assigns confidence, flags specific spans, and calls self-reflection for contradictions.
- Synthesis agent: merges accepted claims, removes disputed spans, resolves false premises, and emits a provenance map for every final sentence.
- Compression agent: invoked by the context manager before budget overflow. It preserves structured artifacts exactly and compresses only conversational filler.
- Meta-agent: reads eval failures and stores a proposed prompt rewrite with a structured diff. It never applies the rewrite automatically.

## Tools And Failure Contracts

The `ToolExecutor` logs every call with input, output, latency, attempt number, and accept/reject status. Rejected outputs are retried up to two times with modified input.

- `web_search_stub`: returns `{chunk_id, url, snippet, relevance_score}`. Failure modes: `timeout`, `empty`, `malformed`.
- `code_execution_sandbox`: runs isolated Python snippets and returns stdout, stderr, and exit code. Failure modes: `timeout`, `malformed`, `error`.
- `structured_data_lookup`: converts natural language to safe SQL over `structured_facts`. Failure modes: `timeout`, `empty`, `malformed`.
- `self_reflection`: rereads session outputs and reports contradictions. Failure modes: `empty`, `malformed`.

Fallback logic is implemented in Python, not prompt text: timeout broadens timeout, empty broadens the query or falls back to another tool, and malformed input is sanitized before retry.

## Evaluation

`app.eval_harness.TEST_CASES` contains 15 cases:

- Five baseline factual/math questions.
- Five ambiguous or underspecified questions.
- Five adversarial questions covering prompt injection, false premises, and contradiction resolution.

Each case receives numeric scores plus justification strings for:

- answer correctness
- citation accuracy
- contradiction resolution quality
- tool selection efficiency
- context budget compliance
- critique agreement with final output

Every eval run stores exact prompts, tool calls, outputs, scores, timestamps, and job traces in the database. Because the system is deterministic, reruns are diff-able.

## Data Handling, Leakage, And Baselines

The local corpus is seeded into `knowledge_chunks` and `structured_facts`. Agents can retrieve from those tables, but eval expectations are only used after a job completes inside scoring code. Prompt rewrites are proposed from failure summaries and require human approval before affecting future runs.

The baseline is a deterministic retrieval-and-lookup system. The extra orchestration complexity is only used where the task requires it: ambiguous decomposition, span-level critique, context-budget policy, and auditable prompt improvement.

## Known Limitations

- The "LLM" behavior is deterministic and rule-based so the assessment runs without paid model credentials. Swapping in a real model would keep the same agent, tool, context, and logging boundaries.
- The Python sandbox uses process isolation and static checks, not a hardened container-per-call jail.
- The search tool is a structured local stub, not internet search.
- Token counting is approximate and intentionally conservative.
- The NL-to-SQL translator supports a safe, small grammar over the seeded fact table.

## What I Would Build Next

- Add real model adapters behind the agent interfaces with JSON schema validation and retryable parse repair.
- Run code execution in a locked-down Firecracker or gVisor sandbox.
- Add migrations and row-level retention policies for production traces.
- Add a small reviewer UI for prompt rewrite diffs, trace replay, and eval regression comparison.
- Split the eval corpus into development, regression, and holdout sets before allowing any approved prompt to graduate.

