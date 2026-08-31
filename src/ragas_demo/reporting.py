import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ragas_demo.evaluation import METRIC_NAMES


def build_report_data(scores: Sequence[dict]) -> dict:
    if not scores:
        raise ValueError("At least one scored case is required")
    aggregates = {}
    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in scores]
        aggregates[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    ranked = []
    for row in scores:
        ranked.append({**row, "metric_mean": statistics.fmean(float(row[m]) for m in METRIC_NAMES)})
    ranked.sort(key=lambda row: (-row["metric_mean"], row["case_id"]))
    strongest = ranked[:2]
    strongest_ids = {row["case_id"] for row in strongest}
    weakest = sorted(
        (row for row in ranked if row["case_id"] not in strongest_ids),
        key=lambda row: (row["metric_mean"], row["case_id"]),
    )[:2]
    return {
        "count": len(scores),
        "aggregates": aggregates,
        "strongest": strongest,
        "weakest": weakest,
    }


def render_markdown(data: dict) -> str:
    lines = [
        "# RAGAS Evaluation Results",
        "",
        (
            "These scores demonstrate the evaluation method on a disposable local corpus; they are "
            "not an evaluation of any production RAG system."
        ),
        "",
        f"Evaluated cases: **{data['count']}**",
        "",
        "## Aggregate scores",
        "",
        "| Metric | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, values in data["aggregates"].items():
        lines.append(
            f"| {metric} | {values['mean']:.3f} | {values['median']:.3f} | "
            f"{values['min']:.3f} | {values['max']:.3f} |"
        )
    example_groups = (
        ("Strongest examples", data["strongest"]),
        ("Weakest examples", data["weakest"]),
    )
    for heading, examples in example_groups:
        lines.extend(["", f"## {heading}", ""])
        for row in examples:
            lines.extend(
                [
                    f"### {row['case_id']} — mean {row['metric_mean']:.3f}",
                    "",
                    f"**Question:** {row['user_input']}",
                    "",
                    f"**Reference:** {row['reference']}",
                    "",
                    f"**Response:** {row['response']}",
                    "",
                    f"**Sources:** {', '.join(row['source_ids']) or 'None'}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_report_artifacts(scores: Sequence[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    data = build_report_data(scores)
    (results_dir / "scores.json").write_text(
        json.dumps(list(scores), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(scores).to_csv(results_dir / "scores.csv", index=False)
    (results_dir / "report.md").write_text(render_markdown(data), encoding="utf-8")
