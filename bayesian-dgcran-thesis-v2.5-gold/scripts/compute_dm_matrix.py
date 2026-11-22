"""
Computes the complete pairwise Diebold-Mariano (Newey-West Bartlett + HLN) statistical significance matrix
across all models in results_dir and exports dm_matrix_<DATASET>.json & .csv for Chapter 4 and thesis.farhadipour.com.
"""
from __future__ import annotations
import os, sys, glob, json, csv
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.metrics.diebold_mariano import diebold_mariano_test


def run_dm_matrix(dataset: str, results_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(results_dir, f"dm_err_{dataset}_*.npy")))
    errs = {}
    for fp in files:
        base = os.path.basename(fp)
        tag = base[len(f"dm_err_{dataset}_") : -4]
        errs[tag] = np.load(fp).astype(np.float64)

    models = list(errs.keys())
    matrix = {}
    rows = []
    for m1 in models:
        matrix[m1] = {}
        for m2 in models:
            if m1 == m2:
                matrix[m1][m2] = {"dm_stat": 0.0, "hln_stat": 0.0, "p_value": 1.0, "significant_05": False}
                continue
            res = diebold_mariano_test(errs[m1], errs[m2], horizon=12)
            dm_s = round(float(res["DM_stat"]), 4)
            hln_s = round(float(res["HLN_stat"]), 4)
            pval = round(float(res["HLN_pvalue_one_sided"]), 6)
            matrix[m1][m2] = {
                "dm_stat": dm_s,
                "hln_stat": hln_s,
                "p_value": pval,
                "significant_05": bool(pval < 0.05),
            }
            rows.append({
                "dataset": dataset, "model_A": m1, "model_B": m2,
                "dm_stat": dm_s, "hln_stat": hln_s, "p_value": pval,
                "significant_05": bool(pval < 0.05),
            })

    out_json = os.path.join(results_dir, f"dm_matrix_{dataset}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "models": models, "matrix": matrix}, f, indent=2)
    out_csv = os.path.join(results_dir, f"dm_matrix_{dataset}.csv")
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Computed {len(models)}x{len(models)} Diebold-Mariano/HLN matrix -> {out_json}")


if __name__ == "__main__":
    run_dm_matrix(sys.argv[1], sys.argv[2])
