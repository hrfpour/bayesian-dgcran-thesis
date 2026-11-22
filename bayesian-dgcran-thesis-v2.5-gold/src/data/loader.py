"""
Unified Raw Traffic Dataset Loader (PeMS03/04/07/08, METR-LA, PEMS-BAY) & PyTorch DataLoaders.
"""
from __future__ import annotations
import os, pickle
from typing import Dict, Any, Tuple
import numpy as np
import yaml

from src.data.preprocessing import (
    StandardScalerMoments,
    impute_train_linear_bounded,
    impute_causal_locf,
    build_calendar_features,
    chronological_split,
    fit_scaler,
    create_sliding_windows,
    build_gaussian_adjacency,
)


def _find_file(data_root: str, target_filename: str) -> str:
    for root, _, files in os.walk(data_root):
        for fn in files:
            if fn.lower() == target_filename.lower():
                return os.path.join(root, fn)
    raise FileNotFoundError(f"Could not find '{target_filename}' under '{data_root}'.")


def load_raw_traffic_and_adj(dataset_name: str, data_root: str, config_path: str = "configs/datasets.yaml") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)["datasets"][dataset_name]

    data_file = _find_file(data_root, spec["raw_files"][0])
    adj_file = _find_file(data_root, spec["raw_files"][1])

    if data_file.lower().endswith(".npz"):
        npz = np.load(data_file)
        key = "data" if "data" in npz else list(npz.keys())[0]
        raw = np.asarray(npz[key], dtype=np.float32)
        if raw.ndim == 2:
            raw = raw[:, :, None]
    elif data_file.lower().endswith(".h5"):
        import pandas as pd
        df = pd.read_hdf(data_file)
        raw = df.values.astype(np.float32)[:, :, None]
    else:
        raise ValueError(f"Unsupported data file format: {data_file}")

    # Load static adjacency
    if adj_file.lower().endswith(".csv"):
        adj = build_gaussian_adjacency(adj_file, num_nodes=spec["num_nodes"])
    elif adj_file.lower().endswith(".pkl"):
        with open(adj_file, "rb") as f:
            try:
                adj_obj = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                adj_obj = pickle.load(f, encoding="latin1")
        adj = np.asarray(adj_obj[2] if isinstance(adj_obj, (list, tuple)) and len(adj_obj) >= 3 else adj_obj, dtype=np.float32)
    else:
        adj = np.eye(spec["num_nodes"], dtype=np.float32)

    return raw, adj, spec


def prepare_dataset_splits(dataset_name: str, data_root: str, config_path: str = "configs/datasets.yaml") -> Dict[str, Any]:
    """
    Returns leak-free preprocessed train/val/test window dicts, train-only scaler, static adj, and spec.
    """
    raw, adj, spec = load_raw_traffic_and_adj(dataset_name, data_root, config_path)
    # Treat exact 0.0 sensor drop-outs in METR-LA/PEMS-BAY speed as missing during imputation only if NaN
    tr, va, te, (t1, t2) = chronological_split(raw, tuple(spec["splits"]))
    tr_imp = impute_train_linear_bounded(tr)
    va_imp = impute_causal_locf(va, tr_imp[-1])
    te_imp = impute_causal_locf(te, va_imp[-1])

    scaler = fit_scaler(tr_imp)
    tr_norm = scaler.transform(tr_imp)
    va_norm = scaler.transform(va_imp)
    te_norm = scaler.transform(te_imp)

    cal_all = build_calendar_features(len(raw))
    cal_tr, cal_va, cal_te = cal_all[:t1], cal_all[t1:t2], cal_all[t2:]

    H, P, ch = spec["history_len"], spec["horizon_len"], spec["target_channel"]
    return {
        "train": create_sliding_windows(tr_norm, cal_tr, H, P, ch),
        "val": create_sliding_windows(va_norm, cal_va, H, P, ch),
        "test": create_sliding_windows(te_norm, cal_te, H, P, ch),
        "scaler": scaler,
        "adj": adj,
        "spec": spec,
    }


def build_torch_dataloaders(splits_dict: Dict[str, Any], batch_size: int = 64, num_workers: int = 2):
    """Creates PyTorch DataLoaders (train shuffled at window level, val/test strictly chronological)."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    loaders = {}
    for split_name in ["train", "val", "test"]:
        d = splits_dict[split_name]
        ds = TensorDataset(
            torch.from_numpy(d["X"]),
            torch.from_numpy(d["T_cal"]),
            torch.from_numpy(d["Y"]),
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == "train"),
        )
    return loaders
