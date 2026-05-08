from app.agents import CompressionAgent
from app.context import ContextBudgetManager, SharedContext


def test_context_compression_preserves_structured_items() -> None:
    context = SharedContext(job_id="job_test", query="hello")
    context.add_item("chat", "filler " * 500, structured=False)
    context.add_tool_result({"tool": "structured_data_lookup", "rows": [{"value": "Paris"}]})
    manager = ContextBudgetManager()
    manager.declare_budget("agent", 80)

    payload = manager.enforce_before_run(context, "agent", CompressionAgent())

    assert context.tool_results[0]["rows"][0]["value"] == "Paris"
    assert payload["tool_results"][0]["rows"][0]["value"] == "Paris"
    assert context.compressed_summary

