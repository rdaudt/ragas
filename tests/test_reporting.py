from ragas_demo.reporting import build_report_data, render_markdown


def score(case_id: str, value: float) -> dict:
    return {
        "case_id": case_id,
        "user_input": f"question {case_id}",
        "reference": f"reference {case_id}",
        "response": f"response {case_id}",
        "source_ids": [f"{case_id}-source"],
        "faithfulness": value,
        "answer_relevancy": value,
        "context_precision": value,
        "context_recall": value,
    }


def test_report_selects_two_strongest_and_two_weakest_without_duplicates() -> None:
    data = build_report_data([score(str(index), index / 10) for index in range(1, 6)])

    assert data["count"] == 5
    assert [row["case_id"] for row in data["strongest"]] == ["5", "4"]
    assert [row["case_id"] for row in data["weakest"]] == ["1", "2"]
    assert data["aggregates"]["faithfulness"]["mean"] == 0.3


def test_markdown_warns_that_scores_are_method_demonstration() -> None:
    markdown = render_markdown(build_report_data([score("a", 0.5)]))

    assert "demonstrate the evaluation method" in markdown
    assert "not an evaluation of any production RAG system" in markdown

