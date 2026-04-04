"""Generate an executive scientific summary for a completed run.

Usage:
    uv run scripts/generate_report.py --run-id run_20260403_2030
"""

import argparse
import csv
import json
from pathlib import Path

from brain_surgery.utils import EXPERIMENTS_DIR, FEATURES_DIR, METRICS_DIR


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if isinstance(loaded, dict):
        return loaded
    return {}


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _specificity_ratio(
    target_delta: float | None, control_delta: float | None
) -> float | None:
    if target_delta is None or control_delta is None or control_delta == 0.0:
        return None
    return target_delta / control_delta


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for run-specific summary generation."""
    parser = argparse.ArgumentParser(
        description="Generate executive summary report for one run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier.")
    return parser.parse_args()


def main() -> None:
    """Aggregate run artifacts into a single executive summary JSON."""
    args = parse_args()
    run_id = args.run_id

    experiment_dir = EXPERIMENTS_DIR / run_id
    features_csv = FEATURES_DIR / run_id / "top_10_features.csv"
    metrics_dbscan_json = METRICS_DIR / run_id / "dbscan_summary.json"
    metrics_elbow_json = METRICS_DIR / "elbow_sweep.json"
    intervention_csv = experiment_dir / "intervention_results.csv"
    metadata_json = experiment_dir / "metadata.json"
    elbow_plot_path = experiment_dir / "plots" / "elbow_method.png"

    top_features = _load_csv_rows(features_csv)
    dbscan_summary = _load_json(metrics_dbscan_json)
    elbow_summary = _load_json(metrics_elbow_json)
    intervention_rows = _load_csv_rows(intervention_csv)
    metadata_summary = _load_json(metadata_json)

    intervention_comparison: list[dict[str, object]] = []
    for row in intervention_rows:
        token = row.get("token", "")
        target_delta = _safe_float(row.get("target_delta"))
        control_delta = _safe_float(row.get("control_delta"))
        ratio = _specificity_ratio(target_delta, control_delta)
        intervention_comparison.append(
            {
                "token": token,
                "target_delta": target_delta,
                "control_delta": control_delta,
                "specificity_score": ratio,
            }
        )

    q6_ronaldo = next(
        (item for item in intervention_comparison if item.get("token") == "Ronaldo"),
        None,
    )

    category_breakdown = {}
    category_obj = metadata_summary.get("category_activation_counts")
    if isinstance(category_obj, dict):
        category_breakdown = category_obj

    report_payload = {
        "run_id": run_id,
        "paths": {
            "experiment_dir": str(experiment_dir),
            "elbow_plot": str(elbow_plot_path),
            "features_csv": str(features_csv),
            "elbow_sweep_json": str(metrics_elbow_json),
            "dbscan_summary_json": str(metrics_dbscan_json),
            "intervention_csv": str(intervention_csv),
            "metadata_json": str(metadata_json),
        },
        "q4_top_features": top_features,
        "q5": {
            "elbow_plot_path": str(elbow_plot_path),
            "elbow_points": elbow_summary.get("elbow_sweep", []),
            "dbscan_cluster_count": dbscan_summary.get("clusters_found"),
        },
        "q6": {
            "ronaldo_vs_neutral": q6_ronaldo,
            "all_tokens": intervention_comparison,
        },
        "metadata_audit": {
            "category_activation_counts": category_breakdown,
        },
    }

    output_path = experiment_dir / "executive_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report_payload, fh, indent=2)

    print(f"Generated executive summary: {output_path}")


if __name__ == "__main__":
    main()
