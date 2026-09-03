import os
import json
import random
import argparse
import csv
import time
import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformer_npi_causal import (
    zscore_per_region,
    compute_fc,
    fc_correlation,
    BrainTransformerCausal,
    BrainTransformerTrainer,
    TransformerNPI,
)


def load_hcp_timeseries(p: str) -> np.ndarray:
    q = Path(p)
    if q.suffix.lower() == '.npy':
        x = np.load(str(q))
    else:
        x = np.loadtxt(str(q))
    if x.ndim == 1:
        x = x[:, None]
    return x


def make_multi_step_windows(x: np.ndarray, steps: int = 3, horizon: int = 50):
    T, N = x.shape
    B = T - steps - horizon + 1
    assert B > 0
    X = np.stack([x[t:t+steps] for t in range(B)], axis=0)
    Y = np.stack([x[t+steps:t+steps+horizon] for t in range(B)], axis=0)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    return X, Y


def make_epoch_history_row(
    epoch: int,
    train_loss: float,
    val_loss: float,
    teacher_forcing: float,
    learning_rate: float,
    best_val: float,
    wait: int,
    epoch_seconds: float,
    improved: bool,
) -> dict:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "teacher_forcing": float(teacher_forcing),
        "learning_rate": float(learning_rate),
        "best_val_loss": float(best_val),
        "early_stop_wait": int(wait),
        "epoch_seconds": float(epoch_seconds),
        "improved": bool(improved),
    }


def summarize_training_history(
    rows: list[dict],
    patience: int,
    stopped_epoch: int | None,
    early_stopped: bool,
) -> dict:
    best_row = min(rows, key=lambda row: row["val_loss"]) if rows else None
    return {
        "n_epochs_completed": len(rows),
        "best_epoch": int(best_row["epoch"]) if best_row else None,
        "best_val_loss": float(best_row["val_loss"]) if best_row else None,
        "stopped_epoch": int(stopped_epoch) if stopped_epoch is not None else None,
        "early_stopped": bool(early_stopped),
        "patience": int(patience),
        "total_training_seconds": float(round(sum(row["epoch_seconds"] for row in rows), 6)),
    }


def write_training_history(out_dir: Path, rows: list[dict], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_history.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "teacher_forcing",
        "learning_rate",
        "best_val_loss",
        "early_stop_wait",
        "epoch_seconds",
        "improved",
    ]
    with (out_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def write_training_history_checkpoint(out_dir: Path, rows: list[dict], patience: int) -> None:
    stopped_epoch = rows[-1]["epoch"] if rows else None
    summary = summarize_training_history(
        rows,
        patience=patience,
        stopped_epoch=stopped_epoch,
        early_stopped=False,
    )
    write_training_history(out_dir, rows, summary)


def parse_args():
    p = argparse.ArgumentParser(description='Transformer NPI Training with Symbol Constraint')
    root = Path(__file__).resolve().parents[2]
    default_data = root / 'NPI-main' / 'NPI-main' / 'real_fMRI_data' / 'sub-102715_bold.npy'
    p.add_argument('--data_path', type=str, default=str(default_data))
    p.add_argument('--seed', type=int, default=20251211)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--horizon', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--fc_loss_weight', type=float, default=0.2)
    p.add_argument('--fc_horizon', type=int, default=50)
    p.add_argument('--fc_roi_sample', type=int, default=0)
    p.add_argument('--fc_amp', action='store_true', default=False)
    p.add_argument('--ms_weight_policy', type=str, default='power')
    p.add_argument('--ms_weight_power', type=float, default=1.5)
    p.add_argument('--sign_loss_weight', type=float, default=0.3,
                   help='Strength of EC sign-consistency regularization (default: 0.3, based on grid search)')
    p.add_argument('--sign_loss_margin', type=float, default=0.8,
                   help='Target dot-product margin for sign consistency (default: 0.8)')
    p.add_argument('--sign_loss_min_amp', type=float, default=0.05,
                   help='Minimum |target| amplitude to include in sign regularization (default: 0.05)')
    p.add_argument('--total_epochs', type=int, default=150)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--tf_anneal_epochs', type=int, default=60)
    p.add_argument('--tf_min', type=float, default=0.1)
    p.add_argument('--noise_std', type=float, default=0.0)
    p.add_argument('--gen_steps', type=int, default=1200)
    p.add_argument('--use_delta', action='store_true', default=True)
    p.add_argument('--run_tag', type=str, default='default')
    p.add_argument('--per_run_zscore', action='store_true', default=False)
    p.add_argument('--fc_eval_seeds', type=int, default=1)
    p.add_argument('--attn', type=str, default='dot', choices=['dot', 'mgka'])
    p.add_argument('--output_base_dir', type=str, default=None,
                   help='Base directory for output (default: validation_results)')
    return p.parse_args()


def _train_eval_core(x: np.ndarray, args, run_tag: str) -> dict:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = Path(__file__).resolve().parents[2]
    per_run = bool(getattr(args, 'per_run_zscore', False))
    if per_run and x.shape[0] % 1200 == 0:
        L = 1200
        R = x.shape[0] // L
        segs = [zscore_per_region(x[i*L:(i+1)*L]) for i in range(R)]
        x = np.concatenate(segs, axis=0)
    else:
        x = zscore_per_region(x)
    T, N = x.shape
    steps = int(args.steps)
    horizon = int(args.horizon)
    X, Y = make_multi_step_windows(x, steps=steps, horizon=horizon)
    B = X.shape[0]
    n_train = int(0.8 * B)
    n_val = int(0.1 * B)
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, B)
    train_ds = data.TensorDataset(X[idx_train], Y[idx_train])
    val_ds = data.TensorDataset(X[idx_val], Y[idx_val])
    test_ds = data.TensorDataset(X[idx_test], Y[idx_test])
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = data.DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, generator=g)
    val_loader = data.DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, generator=g)
    test_loader = data.DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, generator=g)
    model = BrainTransformerCausal(
        n_regions=N,
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        attn_type=str(getattr(args, 'attn', 'dot')),
    )
    trainer = BrainTransformerTrainer(
        model,
        total_epochs=int(args.total_epochs),
        warmup_epochs=int(args.warmup_epochs),
        use_delta=bool(args.use_delta),
        fc_loss_weight=float(args.fc_loss_weight),
        fc_horizon=int(args.fc_horizon),
        fc_roi_sample=int(getattr(args, 'fc_roi_sample', 0)),
        fc_amp=bool(getattr(args, 'fc_amp', False)),
        ms_weight_policy=str(args.ms_weight_policy),
        ms_weight_power=float(args.ms_weight_power),
        sign_loss_weight=float(getattr(args, 'sign_loss_weight', 0.0)),
        sign_loss_margin=float(getattr(args, 'sign_loss_margin', 1.0)),
        sign_loss_min_amp=float(getattr(args, 'sign_loss_min_amp', 0.0)),
    )
    best_val = float('inf')
    wait = 0
    epochs = int(args.total_epochs)
    patience = int(args.patience)
    subject_part = Path(args.data_path).stem.split('_')[0]
    subject_folder = subject_part.replace('sub-', '')
    # 支持自定义输出目录
    if hasattr(args, 'output_base_dir') and args.output_base_dir:
        output_base = Path(args.output_base_dir)
    else:
        output_base = root / 'validation_results'
    out_dir = output_base / subject_folder / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    history_rows = []
    stopped_epoch = None
    early_stopped = False
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        if args.tf_anneal_epochs > 0:
            tf = max(float(args.tf_min), 1.0 - (epoch - 1) / float(args.tf_anneal_epochs))
        else:
            tf = float(args.tf_min)
        trainer.set_teacher_forcing(tf)
        tr_loss = trainer.train_epoch(train_loader)
        prev_tf = trainer.teacher_forcing
        trainer.set_teacher_forcing(0.0)
        va_loss = trainer.validate(val_loader)
        trainer.set_teacher_forcing(prev_tf)
        improved = False
        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            wait = 0
            improved = True
            torch.save(model.state_dict(), str(out_dir / 'best_model.pth'))
        else:
            wait += 1
        row = make_epoch_history_row(
            epoch=epoch,
            train_loss=tr_loss,
            val_loss=va_loss,
            teacher_forcing=tf,
            learning_rate=trainer.opt.param_groups[0]["lr"],
            best_val=best_val,
            wait=wait,
            epoch_seconds=time.time() - epoch_started,
            improved=improved,
        )
        history_rows.append(row)
        write_training_history_checkpoint(out_dir, history_rows, patience=patience)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: train={tr_loss:.4f}, val={va_loss:.4f}, tf={tf:.2f}')
        if wait >= patience:
            print(f'[EarlyStop] at epoch {epoch}, best val={best_val:.4f}')
            stopped_epoch = epoch
            early_stopped = True
            break
    if stopped_epoch is None:
        stopped_epoch = history_rows[-1]["epoch"] if history_rows else None
    training_summary = summarize_training_history(
        history_rows,
        patience=patience,
        stopped_epoch=stopped_epoch,
        early_stopped=early_stopped,
    )
    write_training_history(out_dir, history_rows, training_summary)
    if (out_dir / 'best_model.pth').exists():
        model.load_state_dict(torch.load(str(out_dir / 'best_model.pth'), map_location='cpu'))
    emp_fc = compute_fc(x)
    test_windows = X[idx_test]
    K = max(1, int(getattr(args, 'fc_eval_seeds', 1)))
    num_w = int(test_windows.shape[0])
    if K >= num_w:
        idxs = list(range(num_w))
    else:
        idxs = sorted(set(np.linspace(0, num_w - 1, K, dtype=int).tolist()))
    mod_fcs = []
    for i in idxs:
        seed_window = test_windows[i]
        with torch.no_grad():
            gen = model.generate_recursive(seed_window, steps=int(args.gen_steps), noise_std=float(args.noise_std))
        mod_fcs.append(compute_fc(gen.numpy()))
    if len(mod_fcs) == 1:
        mod_fc = mod_fcs[0]
    else:
        mod_fc = np.mean(np.stack(mod_fcs, axis=0), axis=0)
    r_fc = fc_correlation(emp_fc, mod_fc)
    print('FC reproduction: r = {:.4f}'.format(r_fc))
    test_windows = X[idx_test]
    if len(test_windows) > 256:
        test_windows = test_windows[:256]
    npi = TransformerNPI(model, perturbation=0.05)
    ec = npi.infer_effective_connectivity(test_windows, normalize=True)
    ec_np = ec.cpu().numpy()
    mean = ec_np.mean()
    std = ec_np.std()
    pos_ratio = (ec_np > 0).mean()
    strong_ratio = (np.abs(ec_np) > 0.01).mean()
    print('EC matrix stats: mean={:.6f}, std={:.6f}, max={:.6f}, min={:.6f}'.format(ec_np.mean(), ec_np.std(), ec_np.max(), ec_np.min()))
    print('  positive ratio={:.2f}%, |EC|>0.01 ratio={:.2f}%'.format(pos_ratio*100.0, strong_ratio*100.0))
    np.save(out_dir / 'empirical_fc.npy', emp_fc)
    np.save(out_dir / 'model_fc.npy', mod_fc)
    with open(out_dir / 'fc_r.txt', 'w', encoding='utf-8') as f:
        f.write(f'{r_fc:.6f}')
    np.save(out_dir / 'ec.npy', ec_np)
    with open(out_dir / 'ec_stats.json', 'w', encoding='utf-8') as f:
        json.dump({
            'mean': float(mean),
            'std': float(std),
            'max': float(ec_np.max()),
            'min': float(ec_np.min()),
            'positive_ratio': float(pos_ratio),
            '|EC|>0.01_ratio': float(strong_ratio)
        }, f)
    cfg = vars(args).copy()
    cfg['data_path'] = str(args.data_path)
    with open(out_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # optional: dump biomarkers for MGKA
    if str(getattr(args, 'attn', 'dot')) == 'mgka':
        try:
            bm = model.get_biomarkers()
            with open(out_dir / 'biomarkers.json', 'w', encoding='utf-8') as f:
                json.dump(bm, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return {'best_val': float(best_val), 'r_fc': float(r_fc), 'out_dir': str(out_dir), 'ec': ec_np}


def run_with_array(x: np.ndarray, args, run_tag: str) -> dict:
    return _train_eval_core(np.array(x, copy=True), args, run_tag)


def run_once(args) -> dict:
    data_path = Path(args.data_path)
    assert data_path.exists()
    x = load_hcp_timeseries(str(data_path))
    return _train_eval_core(x, args, args.run_tag)


def main():
    args = parse_args()
    run_once(args)


if __name__ == '__main__':
    main()
