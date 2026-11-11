"""Compare paired LIBERO evaluations without silently mixing conditions."""

from collections import defaultdict
from pathlib import Path

from .common import output_directory, read_json, write_json


def compare_reports(baseline: dict, candidate: dict) -> dict:
    for report in [baseline, candidate]:
        if report.get("status") != "completed":
            raise ValueError("Both evaluations must be completed")
        if not report.get("episodes"):
            raise ValueError("Cannot compare empty evaluations")
    for key in [
        "conditions_sha256",
        "environment_code_sha256",
        "split",
        "assets_revision",
        "versions",
    ]:
        if baseline[key] != candidate[key]:
            raise ValueError(f"Evaluation conditions differ: {key}")

    def indexed(report):
        if any(type(row["success"]) is not bool for row in report["episodes"]):
            raise ValueError("Success values must be booleans")
        rows = {(row["task_id"], row["initial_state"]): row for row in report["episodes"]}
        if len(rows) != len(report["episodes"]):
            raise ValueError("An evaluation contains duplicate cases")
        return rows

    left, right = indexed(baseline), indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError("Evaluations used different cases")
    tasks = defaultdict(lambda: {"trials": 0, "baseline_successes": 0, "candidate_successes": 0})
    pairs = []
    for key, before in left.items():
        after = right[key]
        for field in ["seed", "initial_state_sha256", "instruction"]:
            if before[field] != after[field]:
                raise ValueError(f"Pair {key} differs: {field}")
        entry = tasks[str(key[0])]
        entry["trials"] += 1
        entry["baseline_successes"] += bool(before["success"])
        entry["candidate_successes"] += bool(after["success"])
        pairs.append(
            {
                "task_id": key[0],
                "initial_state": key[1],
                "baseline": bool(before["success"]),
                "candidate": bool(after["success"]),
            }
        )
    n = len(pairs)
    baseline_rate = sum(p["baseline"] for p in pairs) / n
    candidate_rate = sum(p["candidate"] for p in pairs) / n
    return {
        "status": "completed",
        "split": baseline["split"],
        "trials": n,
        "baseline_success_rate": baseline_rate,
        "candidate_success_rate": candidate_rate,
        "difference_percentage_points": 100 * (candidate_rate - baseline_rate),
        "task_results": dict(tasks),
        "pairs": pairs,
        "note": "Paired descriptive comparison; small samples do not establish generalization",
    }


def compare(baseline: str, candidate: str) -> Path:
    left = read_json(Path(baseline) / "report.json")
    right = read_json(Path(candidate) / "report.json")
    result = compare_reports(left, right)
    result.update(baseline_directory=baseline, candidate_directory=candidate)
    folder = output_directory("comparison")
    write_json(folder / "comparison.json", result)
    print(f"Results: {folder}", flush=True)
    return folder
