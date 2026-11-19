"""
Automatic Literature Verification Script.
Compares our Kaggle-reproduced metrics against verified published figures in configs/published_benchmarks.yaml
and exports a clean JSON + CSV verification report for Chapter 4 and thesis.farhadipour.com.
"""
from __future__ import annotations
import os, sys, glob, json, csv
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def compare_results(dataset: str, results_dir: str) -> None:
    pub_path = os.path.join(REPO_ROOT, "configs", "published_benchmarks.yaml")
    with open(pub_path, "r", encoding="utf-8") as f:
        pub = yaml.safe_load(f).get(dataset, {}).get("models", {})

    found = {}
    for fp in sorted(glob.glob(os.path.join(results_dir, f"{dataset}_*_seed*.json"))):
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        found[d["model"]] = d

    print(f"\n================ Literature Verification Report: {dataset} ================")
    header = f"{'Model':18s} | {'Ours MAE':>9s} | {'Paper MAE':>9s} | {'Diff%':>7s} | {'Ours RMSE':>9s} | {'Paper RMSE':>10s} | Source"
    print(header)
    print("-" * len(header))

    rows = []
    for model_name, rec in found.items():
        ov = rec["metrics"]["overall"]
        ours_mae = ov.get("MAE")
        ours_rmse = ov.get("RMSE")
        ours_mape = ov.get("MAPE")
        ref = pub.get(model_name, {})
        paper_mae = ref.get("mae", ref.get("h12_mae"))
        paper_rmse = ref.get("rmse", ref.get("h12_rmse"))
        src = ref.get("source", "—")
        diff_pct = ((ours_mae - paper_mae) / paper_mae * 100.0) if (ours_mae is not None and paper_mae) else None
        p_mae_s = f"{paper_mae:9.2f}" if paper_mae is not None else f"{'—':>9s}"
        p_rmse_s = f"{paper_rmse:10.2f}" if paper_rmse is not None else f"{'—':>10s}"
        diff_s = f"{diff_pct:+6.1f}%" if diff_pct is not None else f"{'—':>7s}"
        print(f"{model_name:18s} | {ours_mae:9.2f} | {p_mae_s} | {diff_s} | {ours_rmse:9.2f} | {p_rmse_s} | {src}")
        rows.append({
            "dataset": dataset, "model": model_name,
            "ours_mae": ours_mae, "paper_mae": paper_mae, "diff_mae_pct": round(diff_pct, 2) if diff_pct is not None else None,
            "ours_rmse": ours_rmse, "paper_rmse": paper_rmse, "ours_mape": ours_mape,
            "paper_source": src,
        })

    out_json = os.path.join(results_dir, f"literature_verification_{dataset}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    out_csv = os.path.join(results_dir, f"literature_verification_{dataset}.csv")
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nSaved verification report -> {out_json}")


if __name__ == "__main__":
    ds = sys.argv[1]
    rdir = sys.argv[2]
    compare_results(ds, rdir)
