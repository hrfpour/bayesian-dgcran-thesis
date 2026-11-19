"""
Standardized JSON/CSV Exporter for Chapter 4 LaTeX Tables and thesis.farhadipour.com.
"""
from __future__ import annotations
import json, os, csv
from datetime import datetime, timezone
from typing import Dict, Any


def save_experiment_result(
    output_dir: str,
    dataset: str,
    model_name: str,
    metrics: Dict[str, Any],
    hyperparams: Dict[str, Any],
    dm_tests: Dict[str, Any] | None = None,
    seed: int = 42,
) -> str:
    """Saves structured JSON (consumed directly by thesis.farhadipour.com) and flat summary CSV."""
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "hyperparams": hyperparams,
        "metrics": metrics,
        "dm_tests": dm_tests or {},
    }
    json_path = os.path.join(output_dir, f"{dataset}_{model_name}_seed{seed}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "summary_leaderboard.csv")
    ov = metrics.get("overall", {})
    row = {"dataset": dataset, "model": model_name, "seed": seed}
    for k in ["MAE", "RMSE", "MAPE", "PICP", "MPIW", "MHPICE", "MIS", "MNLL", "CRPS_Gaussian", "CRPS_Empirical", "ECE"]:
        if k in ov:
            row[k] = round(float(ov[k]), 5)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return json_path
