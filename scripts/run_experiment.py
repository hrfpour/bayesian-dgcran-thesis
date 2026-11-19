"""
Main CLI Runner for Training, Ablation, Robustness & Benchmarking on Kaggle GPU.
Example:
  python3 scripts/run_experiment.py --dataset PeMS08 --model Bayesian-DGCRAN --data_root /kaggle/input/pems-metr-traffic-benchmarks
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch

from src.data.loader import prepare_dataset_splits, build_torch_dataloaders
from src.data.preprocessing import inject_missing_and_noise
from src.models.baselines import HistoricalAverageBaseline, RidgeVARBaseline, build_model
from src.engine.trainer import train_single_model, run_inference_split, evaluate_probabilistic_with_calibration
from src.metrics.evaluation import compute_point_metrics
from src.utils.io_schema import save_experiment_result


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["PeMS03", "PeMS04", "PeMS07", "PeMS08", "METR-LA", "PEMS-BAY"])
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data_root", type=str, default="/kaggle/input")
    p.add_argument("--output_dir", type=str, default="results/web_export")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--awa_epochs", type=int, default=20)
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

    # Optional test-time corruption for Chapter 4 Robustness Study
    if args.missing_rate > 0.0 or args.noise_std > 0.0:
        splits["test"]["X"] = inject_missing_and_noise(
            splits["test"]["X"], missing_rate=args.missing_rate, noise_std=args.noise_std, seed=args.seed
        )

    # Non-PyTorch Statistical Baselines (HA, VAR)
    if args.model in ("HA", "VAR"):
        y_te_phys = scaler.inverse_mean(splits["test"]["Y"], ch)
        if args.model == "HA":
            pred_norm = HistoricalAverageBaseline().predict(splits["test"]["X"], spec["horizon_len"], ch)
        else:
            var_m = RidgeVARBaseline(alpha=1.0).fit(splits["train"]["X"], splits["train"]["Y"], ch)
            pred_norm = var_m.predict(splits["test"]["X"], ch)
        pred_phys = scaler.inverse_mean(pred_norm, ch)
        metrics = {"overall": compute_point_metrics(y_te_phys, pred_phys)}
        save_experiment_result(args.output_dir, args.dataset, args.model, metrics, vars(args), seed=args.seed)
        print(f"[{args.dataset} | {args.model}] Done:", metrics["overall"])
        return

    loaders = build_torch_dataloaders(splits, batch_size=spec["batch_size"])
    model = build_model(args.model, spec, splits["adj"], hidden_dim=args.hidden_dim)

    is_det = args.model in ("DCRNN", "AGCRN", "DGCRN", "DGCRAN")
    use_kl_awa = args.model not in ("Ablation-NoKLAWA",) and (not is_det)
    use_nemhcc = args.model not in ("Ablation-NoNEMHCC",)

    t0 = time.time()
    if args.model == "DeepEnsembles":
        for idx, member in enumerate(model.members):
            set_seed(args.seed + idx * 101)
            train_single_model(member, loaders, device, args.epochs, 0, 0, use_amp=spec["use_amp"], deterministic_mae_only=False, use_kl_awa=False)
        model = model.to(device)
    else:
        model = train_single_model(
            model, loaders, device,
            epochs_pre=args.epochs, epochs_warmup=args.warmup_epochs, epochs_awa=args.awa_epochs,
            use_amp=spec["use_amp"], deterministic_mae_only=is_det, use_kl_awa=use_kl_awa,
        )
    train_sec = round(time.time() - t0, 2)

    val_out = run_inference_split(model, loaders["val"], scaler, device, ch, args.num_mc, is_probabilistic=(not is_det))
    test_out = run_inference_split(model, loaders["test"], scaler, device, ch, args.num_mc, is_probabilistic=(not is_det))

    if is_det:
        metrics = {"overall": compute_point_metrics(test_out["y_true"], test_out["mu"])}
    else:
        # DeepSTUQ uses exchangeable MHCC (rho_cal = 1.0); Bayesian-DGCRAN uses non-exchangeable rho_cal = 0.995
        rho_cal = 1.0 if args.model == "DeepSTUQ" else 0.995
        metrics = evaluate_probabilistic_with_calibration(val_out, test_out, use_nemhcc=use_nemhcc, rho_cal=rho_cal)

    hp = dict(vars(args))
    hp["train_seconds"] = train_sec
    tag = args.model if (args.missing_rate == 0 and args.noise_std == 0) else f"{args.model}_miss{int(args.missing_rate*100)}_noise{args.noise_std}"
    out_json = save_experiment_result(args.output_dir, args.dataset, tag, metrics, hp, seed=args.seed)
    print(f"[{args.dataset} | {tag}] Finished in {train_sec}s -> {out_json}")
    print("  Overall metrics:", metrics["overall"])


if __name__ == "__main__":
    main()
