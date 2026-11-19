"""
Dataset Verification & Normalization Script for Kaggle / Local Execution.
Checks all 6 thesis datasets (PeMS03, PeMS04, PeMS07, PeMS08, METR-LA, PEMS-BAY) against Table 3.4 specs.
"""
from __future__ import annotations
import os, sys
import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def verify_dataset_folder(data_root: str, config_path: str = "configs/datasets.yaml") -> None:
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["datasets"]

    print(f"Checking all {len(cfg)} thesis datasets under: {data_root}\n")
    for name, spec in cfg.items():
        found = []
        for root, _, files in os.walk(data_root):
            for fn in files:
                if fn.lower() in [r.lower() for r in spec["raw_files"]]:
                    found.append(os.path.join(root, fn))
        status = "OK" if len(found) == len(spec["raw_files"]) else f"MISSING ({len(found)}/{len(spec['raw_files'])})"
        print(f"  [{name:8s}] expected N={spec['num_nodes']:3d}, T={spec['num_timesteps']:5d} -> {status}")
        for p in sorted(found):
            if p.lower().endswith(".npz"):
                arr = np.load(p)
                key = "data" if "data" in arr else list(arr.keys())[0]
                print(f"             NPZ {os.path.basename(p)}: shape={arr[key].shape}, dtype={arr[key].dtype}")
            elif p.lower().endswith(".h5"):
                size_mb = os.path.getsize(p) / (1024 * 1024)
                print(f"             H5  {os.path.basename(p)}: {size_mb:.1f} MB")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input"
    verify_dataset_folder(root)
