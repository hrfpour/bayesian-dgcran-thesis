"""
Main CLI Runner (v2.2 Gold) for Training, Complete Metric Calculation, Spatial/Trajectory Export & DM Vectors.
"""
from __future__ import annotations
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import argparse, json, os, time
import numpy as np
import torch

from src.data.loader import prepare_dataset_splits, build_torch_dataloaders
from src.data.preprocessing import inject_missing_and_noise
from src.models.baselines import HistoricalAverageBaseline, RidgeVARBaseline, FastSARIMABaseline, build_model
from src.engine.trainer import train_single_model, run_inference_split, evaluate_probabilistic_with_calibration
from src.metrics.evaluation import compute_point_metrics
from src.utils.io_schema import save_experiment_result


DETERMINISTIC_NEURAL = {
    "FC-LSTM", "STGCN", "DCRNN", "ASTGCN", "GraphWaveNet", "AGCRN", "DGCRN", "STAEformer", "DGCRAN"
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_horizon_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    horizons = {}
    for h_idx in range(y_true.shape[1]):
        horizons[f"h{h_idx + 1}"] = compute_point_metrics(y_true[:, h_idx : h_idx + 1, :], y_pred[:, h_idx : h_idx + 1, :])
    return horizons


def _export_spatial_and_trajectories(output_dir: str, dataset: str, tag: str, y_true: np.ndarray, mu: np.ndarray,
                                     var_alea: np.ndarray | None = None, var_epis: np.ndarray | None = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    err = y_true - mu
    sensor_mae = np.mean(np.abs(err), axis=(0, 1))
    sensor_rmse = np.sqrt(np.mean(err ** 2, axis=(0, 1)))
    spatial_payload = {
        "dataset": dataset,
        "model": tag,
        "num_nodes": int(y_true.shape[2]),
        "sensor_mae": np.round(sensor_mae.astype(float), 3).tolist(),
        "sensor_rmse": np.round(sensor_rmse.astype(float), 3).tolist(),
    }
    with open(os.path.join(output_dir, f"spatial_{dataset}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(spatial_payload, f)

    M_s = min(144, y_true.shape[0])
    N = y_true.shape[2]
    sensor_ids = sorted(list({int(round(i)) for i in np.linspace(0, N - 1, min(5, N))}))
    traj_payload = {"dataset": dataset, "model": tag, "steps": M_s, "horizons": {}}
    for h_step in [1, 3, 6, 12]:
        if h_step > y_true.shape[1]:
            continue
        hi = h_step - 1
        h_dict = {}
        for sid in sensor_ids:
            entry = {
                "y_true": np.round(y_true[:M_s, hi, sid].astype(float), 2).tolist(),
                "mu": np.round(mu[:M_s, hi, sid].astype(float), 2).tolist(),
            }
            if var_alea is not None and var_epis is not None and np.any(var_alea > 0):
                std_tot = np.sqrt(np.maximum(var_alea[:M_s, hi, sid] + var_epis[:M_s, hi, sid], 1e-8))
                entry["std_alea"] = np.round(np.sqrt(np.maximum(var_alea[:M_s, hi, sid], 1e-8)).astype(float), 2).tolist()
                entry["std_epis"] = np.round(np.sqrt(np.maximum(var_epis[:M_s, hi, sid], 1e-8)).astype(float), 2).tolist()
                entry["lower_90"] = np.round((mu[:M_s, hi, sid] - 1.645 * std_tot).astype(float), 2).tolist()
                entry["upper_90"] = np.round((mu[:M_s, hi, sid] + 1.645 * std_tot).astype(float), 2).tolist()
            h_dict[str(sid)] = entry
        traj_payload["horizons"][f"h{h_step}"] = h_dict
    with open(os.path.join(output_dir, f"traj_{dataset}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(traj_payload, f)

    mae_per_sample = np.mean(np.abs(err), axis=(1, 2)).astype(np.float32)
    np.save(os.path.join(output_dir, f"dm_err_{dataset}_{tag}.npy"), mae_per_sample)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["PeMS03", "PeMS04", "PeMS07", "PeMS08", "METR-LA", "PEMS-BAY"])
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data_root", type=str, default="/kaggle/input/datasets/hrfp1234/pems-metr-traffic-benchmarks")
    p.add_argument("--output_dir", type=str, default="results/web_export")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--awa_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_mc", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--missing_rate", type=float, default=0.0)
    p.add_argument("--noise_std", type=float, default=0.0)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = prepare_dataset_splits(args.dataset, args.data_root)
    spec = splits["spec"]
    scaler = splits["scaler"]
    ch = spec["target_channel"]

    if args.missing_rate > 0.0 or args.noise_std > 0.0:
        splits["test"]["X"] = inject_missing_and_noise(
            splits["test"]["X"], missing_rate=args.missing_rate, noise_std=args.noise_std, seed=args.seed
        )

    if args.model in ("HA", "VAR", "SARIMA"):
        t0 = time.time()
        y_te_phys = scaler.inverse_mean(splits["test"]["Y"], ch)
        if args.model == "HA":
            pred_norm = HistoricalAverageBaseline().predict(splits["test"]["X"], spec["horizon_len"], ch)
        elif args.model == "VAR":
            var_m = RidgeVARBaseline(alpha=1.0).fit(splits["train"]["X"], splits["train"]["Y"], ch)
            pred_norm = var_m.predict(splits["test"]["X"], ch)
        else:
            sar_m = FastSARIMABaseline(steps_per_day=288, alpha=1.0).fit(
                splits["train"]["X"], splits["train"]["T_cal"], splits["train"]["Y"], ch
            )
            pred_norm = sar_m.predict(splits["test"]["X"], splits["test"]["T_cal"], ch)
        pred_phys = scaler.inverse_mean(pred_norm, ch)
        metrics = {
            "overall": compute_point_metrics(y_te_phys, pred_phys),
            "horizons": _compute_horizon_breakdown(y_te_phys, pred_phys),
        }
        hp = dict(vars(args))
        hp["train_seconds"] = round(time.time() - t0, 2)
        hp["num_parameters"] = 0
        hp["inference_ms_per_batch"] = 0.1
        out_json = save_experiment_result(args.output_dir, args.dataset, args.model, metrics, hp, seed=args.seed)
        _export_spatial_and_trajectories(args.output_dir, args.dataset, args.model, y_te_phys, pred_phys)
        print(f"[{args.dataset} | {args.model}] Done in {hp['train_seconds']}s -> {out_json}")
        print("  Overall:", metrics["overall"])
        return

    loaders = build_torch_dataloaders(splits, batch_size=spec["batch_size"])
    model = build_model(args.model, spec, splits["adj"], hidden_dim=args.hidden_dim)

    is_det = args.model in DETERMINISTIC_NEURAL
    use_kl_awa = (args.model not in ("Ablation-NoKLAWA",)) and (not is_det)
    use_nemhcc = args.model not in ("Ablation-NoNEMHCC",)

    t0 = time.time()
    history = {}
    if args.model == "DeepEnsembles":
        for idx, member in enumerate(model.members):
            set_seed(args.seed + idx * 101)
            _, hist_m = train_single_model(
                member, loaders, device, args.epochs, 0, 0,
                use_amp=spec["use_amp"], deterministic_mae_only=False, use_kl_awa=False, patience=args.patience
            )
            if idx == 0:
                history = hist_m
        model = model.to(device)
    else:
        model, history = train_single_model(
            model, loaders, device,
            epochs_pre=args.epochs, epochs_warmup=args.warmup_epochs, epochs_awa=args.awa_epochs,
            use_amp=spec["use_amp"], deterministic_mae_only=is_det, use_kl_awa=use_kl_awa, patience=args.patience,
        )
    train_sec = round(time.time() - t0, 2)

    val_out = run_inference_split(model, loaders["val"], scaler, device, ch, args.num_mc, is_probabilistic=(not is_det), use_amp=spec["use_amp"])
    test_out = run_inference_split(model, loaders["test"], scaler, device, ch, args.num_mc, is_probabilistic=(not is_det), use_amp=spec["use_amp"])

    if is_det:
        metrics = {
            "overall": compute_point_metrics(test_out["y_true"], test_out["mu"]),
            "horizons": _compute_horizon_breakdown(test_out["y_true"], test_out["mu"]),
            "learning_curve": history,
        }
    else:
        rho_cal = 1.0 if args.model == "DeepSTUQ" else 0.995
        metrics = evaluate_probabilistic_with_calibration(val_out, test_out, use_nemhcc=use_nemhcc, rho_cal=rho_cal)
        metrics["horizons"] = _compute_horizon_breakdown(test_out["y_true"], test_out["mu"])
        metrics["learning_curve"] = history

    hp = dict(vars(args))
    hp["train_seconds"] = train_sec
    hp["num_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    hp["inference_ms_per_batch"] = test_out["inference_ms_per_batch"]
    tag = args.model if (args.missing_rate == 0 and args.noise_std == 0) else f"{args.model}_miss{int(args.missing_rate*100)}_noise{args.noise_std}"
    out_json = save_experiment_result(args.output_dir, args.dataset, tag, metrics, hp, seed=args.seed)
    _export_spatial_and_trajectories(
        args.output_dir, args.dataset, tag, test_out["y_true"], test_out["mu"],
        test_out.get("var_alea"), test_out.get("var_epis")
    )

    print(f"[{args.dataset} | {tag}] Finished in {train_sec}s (ep={history.get('epochs_run')}, best={history.get('best_epoch')}, params={hp['num_parameters']:,}, inf={hp['inference_ms_per_batch']}ms) -> {out_json}")
    print("  Overall:", metrics["overall"])


if __name__ == "__main__":
    main()
