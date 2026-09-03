import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def zscore_per_region(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sigma


def make_sliding_windows(x: np.ndarray, steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    T, N = x.shape
    X = []
    Y = []
    for t in range(T - steps):
        X.append(x[t:t+steps])
        Y.append(x[t+steps])
    X = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y, axis=0), dtype=torch.float32)
    return X, Y


def compute_fc(x: np.ndarray) -> np.ndarray:
    # Robust FC: sanitize and compute Pearson via z-scoring to avoid NaNs
    x = np.asarray(x)
    # Replace NaN/Inf that may arise from unstable generation
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 2 or x.shape[0] < 2:
        N = x.shape[1] if x.ndim == 2 else 0
        return np.eye(N, dtype=float)
    # z-score per region with epsilon
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-8
    z = (x - mu) / sigma
    # Correlation as normalized covariance
    T = max(1, z.shape[0] - 1)
    fc = (z.T @ z) / T
    # Numerical safety
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(fc, 1.0)
    return fc


def fc_correlation(empirical_fc: np.ndarray, model_fc: np.ndarray) -> float:
    iu = np.triu_indices_from(empirical_fc, k=1)
    e = empirical_fc[iu]
    m = model_fc[iu]
    # mask invalid entries
    mask = np.isfinite(e) & np.isfinite(m)
    if mask.sum() < 10:
        return 0.0
    e = e[mask]
    m = m[mask]
    if e.std() < 1e-8 or m.std() < 1e-8:
        return 0.0
    r = np.corrcoef(e, m)[0, 1]
    if not np.isfinite(r):
        return 0.0
    return float(r)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


class MultiScaleGaussianKernelAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # initialize beta ~ sqrt(head_dim) to avoid overly sharp kernels at start
        self.log_beta = nn.Parameter(torch.full((n_heads,), math.log(max(1e-6, math.sqrt(self.head_dim)))))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        H = self.n_heads
        hd = self.head_dim
        xh = x.view(B, S, H, hd).transpose(1, 2)
        xi = xh.unsqueeze(3)
        xj = xh.unsqueeze(2)
        diff = xi - xj
        # scale-invariant distance across head dimension
        dist2 = (diff * diff).sum(-1) / float(max(1, hd))
        beta = torch.exp(self.log_beta).view(1, H, 1, 1)
        score = -dist2 / (2.0 * (beta * beta + 1e-8))
        if attn_mask is not None:
            score = score + attn_mask.view(1, 1, S, S)
        if key_padding_mask is not None:
            mask = key_padding_mask.view(B, 1, 1, S).bool()
            score = score.masked_fill(mask, float('-inf'))
        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, xh)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        return out


class MGKALayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiScaleGaussianKernelAttention(d_model, n_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        sa = self.attn(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.drop1(sa)
        h2 = self.norm2(x)
        ff = self.ff(h2)
        x = x + self.drop2(ff)
        return x


class MGKAEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([MGKALayer(d_model, n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return h

    def get_betas(self) -> List[torch.Tensor]:
        betas = []
        for layer in self.layers:
            betas.append(torch.exp(layer.attn.log_beta).detach())
        return betas

class BrainTransformerCausal(nn.Module):
    def __init__(self, n_regions: int, d_model: int = 256, n_heads: int = 4, n_layers: int = 4, d_ff: int = 1024, dropout: float = 0.1, max_seq_len: int = 32, attn_type: str = 'dot'):
        super().__init__()
        self.n_regions = n_regions
        self.d_model = d_model
        self.input_proj = nn.Linear(n_regions, d_model)
        self.attn_type = str(attn_type)
        if self.attn_type == 'mgka':
            self.encoder = MGKAEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout)
        else:
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos = PositionalEncoding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(d_model, n_regions)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, steps, N] -> predict next step [B, N]
        B, S, _ = x.shape
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.dropout(h)
        attn_mask = build_causal_mask(S, x.device)
        if self.attn_type == 'mgka':
            h = self.encoder(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        else:
            h = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        last = h[:, -1, :]
        out = self.readout(last)
        return out

    @torch.no_grad()
    def get_biomarkers(self):
        if self.attn_type == 'mgka' and hasattr(self.encoder, 'get_betas'):
            betas = [b.cpu().numpy() for b in self.encoder.get_betas()]
            return {'attn_type': 'mgka', 'betas_per_layer': [b.tolist() for b in betas]}
        return {'attn_type': self.attn_type}

    @torch.no_grad()
    def generate_recursive(self, seed_window: torch.Tensor, steps: int = 1200, noise_std: float = 0.1) -> torch.Tensor:
        # seed_window: [steps0, N]
        self.eval()
        device = next(self.parameters()).device
        win = seed_window.clone().to(device)
        hist = [win.detach().cpu().numpy()]
        S0, N = win.shape
        for _ in range(steps):
            inp = win.unsqueeze(0)  # [1,S0,N]
            pred = self.forward(inp)  # [1,N]
            pred = pred.squeeze(0)
            if noise_std > 0:
                pred = pred + noise_std * torch.randn_like(pred)
            # Stabilize: remove NaN/Inf and clip to a reasonable range
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            pred = pred.clamp_(-5.0, 5.0)
            win = torch.cat([win[1:], pred.unsqueeze(0)], dim=0)
            hist.append(pred.detach().cpu().numpy()[None, :])
        out = np.concatenate(hist, axis=0)  # [S0 + steps, N]
        return torch.tensor(out, dtype=torch.float32)


class BrainTransformerTrainer:
    def __init__(self, model: BrainTransformerCausal, lr: float = 1e-3, weight_decay: float = 1e-4, device: Optional[str] = None,
                 total_epochs: int = 100, warmup_epochs: int = 0, use_delta: bool = False,
                 fc_loss_weight: float = 0.0, fc_horizon: int = 20,
                 ms_weight_policy: str = 'uniform', ms_weight_power: float = 1.0,
                 fc_roi_sample: int = 0, fc_amp: bool = False,
                 sign_loss_weight: float = 0.0, sign_loss_margin: float = 1.0,
                 sign_loss_min_amp: float = 0.0):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.crit = nn.MSELoss()
        # learning rate schedule: linear warmup then cosine to 0
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        def lr_lambda(epoch: int):
            e = float(epoch)
            if self.warmup_epochs > 0 and e < self.warmup_epochs:
                return (e + 1.0) / float(self.warmup_epochs)
            # cosine over remaining epochs
            rem = max(1.0, float(self.total_epochs - self.warmup_epochs))
            progress = min(1.0, max(0.0, (e - self.warmup_epochs) / rem))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        # scheduled sampling: default teacher forcing prob
        self.teacher_forcing: float = 1.0
        # whether to train on deltas (next - last)
        self.use_delta: bool = bool(use_delta)
        # auxiliary FC loss controls
        self.fc_loss_weight: float = float(fc_loss_weight)
        self.fc_horizon: int = int(max(2, fc_horizon))
        # multi-step loss weighting
        self.ms_weight_policy: str = str(ms_weight_policy)
        self.ms_weight_power: float = float(ms_weight_power)
        self.fc_roi_sample: int = int(max(0, fc_roi_sample))
        self.fc_amp: bool = bool(fc_amp)
        self.sign_loss_weight: float = float(max(0.0, sign_loss_weight))
        self.sign_loss_margin: float = float(max(1e-6, sign_loss_margin))
        self.sign_loss_min_amp: float = float(max(0.0, sign_loss_min_amp))

    @staticmethod
    def _fc_from_sequence(seq: torch.Tensor) -> torch.Tensor:
        """Compute per-batch correlation-like FC from sequence [B,T,N]."""
        # z-score along time dim
        mu = seq.mean(dim=1, keepdim=True)
        std = seq.std(dim=1, unbiased=False, keepdim=True) + 1e-6
        z = (seq - mu) / std
        # FC = Z^T Z / (T-1)
        B, T, N = z.shape
        fc = torch.matmul(z.transpose(1, 2), z) / max(1, (T - 1))  # [B,N,N]
        return fc

    def set_teacher_forcing(self, p: float) -> None:
        self.teacher_forcing = float(max(0.0, min(1.0, p)))

    def set_fc_weight(self, w: float) -> None:
        self.fc_loss_weight = float(max(0.0, w))

    def set_sign_loss(self, weight: float, margin: Optional[float] = None, min_amp: Optional[float] = None) -> None:
        self.sign_loss_weight = float(max(0.0, weight))
        if margin is not None:
            self.sign_loss_margin = float(max(1e-6, margin))
        if min_amp is not None:
            self.sign_loss_min_amp = float(max(0.0, min_amp))

    def _sign_penalty(self, pred_abs: torch.Tensor, target_abs: torch.Tensor) -> torch.Tensor:
        """Compute smooth penalty encouraging predicted EC to share sign with targets.

        Args:
            pred_abs: absolute predictions [B, N]
            target_abs: absolute targets [B, N]

        Returns:
            Scalar penalty tensor.
        """
        if self.sign_loss_weight <= 0.0:
            return torch.tensor(0.0, device=pred_abs.device)
        # Focus on entries with sufficient magnitude in targets
        tgt_mag = target_abs.abs()
        mask = tgt_mag >= self.sign_loss_min_amp
        if not mask.any():
            return torch.tensor(0.0, device=pred_abs.device)
        pred_sel = pred_abs[mask]
        target_sel = target_abs[mask]
        # encourage pred * target to be large and positive
        prod = pred_sel * target_sel
        # hinge-like smooth penalty: relu(margin - prod)
        margin = self.sign_loss_margin
        penalty = torch.relu(margin - prod)
        return penalty.mean() if penalty.numel() > 0 else torch.tensor(0.0, device=pred_abs.device)

    def _step(self, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        xb = xb.to(self.device)
        yb = yb.to(self.device)
        use_amp = self.device.type == 'cuda'
        with torch.amp.autocast('cuda', enabled=use_amp):
            # If yb is [B,N]: one-step loss as原
            if yb.dim() == 2:
                pred = self.model(xb)  # [B,N] absolute next
                if self.use_delta:
                    last = xb[:, -1, :]
                    loss = self.crit(pred - last, yb - last)
                    pred_abs = last + (pred - last)
                else:
                    loss = self.crit(pred, yb)
                    pred_abs = pred
                if self.sign_loss_weight > 0.0:
                    sign_penalty = self._sign_penalty(pred_abs, yb)
                    loss = loss + self.sign_loss_weight * sign_penalty
                return loss
            # If yb is [B,K,N]: multi-step rollout with own predictions
            # Roll the input window and accumulate MSE across K steps
            elif yb.dim() == 3:
                B, K, N = yb.shape
                win = xb.clone()
                # prepare per-step weights
                if K <= 0:
                    return torch.tensor(0.0, device=xb.device)
                idx = torch.arange(1, K + 1, device=xb.device).float()
                if self.ms_weight_policy == 'linear':
                    w = idx
                elif self.ms_weight_policy == 'power':
                    p = max(0.0, float(self.ms_weight_power))
                    w = idx.pow(p)
                else:
                    # uniform
                    w = torch.ones_like(idx)
                w = w / (w.sum() + 1e-8)
                loss_main = torch.tensor(0.0, device=xb.device)
                sign_loss_acc = torch.tensor(0.0, device=xb.device)
                collect_fc = (self.fc_loss_weight > 0.0)
                pred_steps = [] if collect_fc else None  # collect absolute predictions for FC auxiliary loss
                for s in range(K):
                    pred = self.model(win)  # [B,N] absolute next
                    target_next = yb[:, s, :]  # absolute next
                    last = win[:, -1, :]
                    # absolute prediction used for rollout and FC loss
                    pred_abs = pred if not self.use_delta else (last + (pred - last))
                    # step loss
                    if self.use_delta:
                        step_loss = self.crit(pred - last, target_next - last)
                    else:
                        step_loss = self.crit(pred, target_next)
                    # scheduled sampling: choose next absolute frame for window update
                    if self.teacher_forcing >= 1.0:
                        next_abs = target_next
                    elif self.teacher_forcing <= 0.0:
                        next_abs = pred_abs
                    else:
                        # mask shape [B,1] to choose per-sample
                        m = (torch.rand(B, 1, device=win.device) < self.teacher_forcing).float()
                        next_abs = m * target_next + (1.0 - m) * pred_abs
                    next_exp = next_abs.unsqueeze(1)  # [B,1,N]
                    win = torch.cat([win[:, 1:, :], next_exp], dim=1)
                    # collect absolute predictions for FC auxiliary loss
                    if collect_fc:
                        pred_steps.append(pred_abs.detach() if not self.model.training else pred_abs)
                    # accumulate weighted loss
                    loss_main = loss_main + w[s] * step_loss
                    if self.sign_loss_weight > 0.0:
                        sign_penalty = self._sign_penalty(pred_abs, target_next)
                        sign_loss_acc = sign_loss_acc + w[s] * sign_penalty
                loss_total = loss_main
                if self.sign_loss_weight > 0.0:
                    loss_total = loss_total + self.sign_loss_weight * sign_loss_acc
                if collect_fc and len(pred_steps) >= 2:
                    Tfc = min(len(pred_steps), self.fc_horizon)
                    pred_seq = torch.stack(pred_steps[:Tfc], dim=1)
                    true_seq = yb[:, :Tfc, :]
                    Nloc_all = pred_seq.size(-1)
                    if self.fc_roi_sample > 0 and self.fc_roi_sample < Nloc_all:
                        m = int(self.fc_roi_sample)
                        idx_roi = torch.randperm(Nloc_all, device=pred_seq.device)[:m]
                        pred_seq = pred_seq.index_select(dim=2, index=idx_roi)
                        true_seq = true_seq.index_select(dim=2, index=idx_roi)
                    fc_pred = self._fc_from_sequence(pred_seq)
                    with torch.no_grad():
                        fc_true = self._fc_from_sequence(true_seq)
                    # upper-triangular MSE
                    Nloc = fc_pred.size(-1)
                    mask = torch.triu(torch.ones(Nloc, Nloc, device=fc_pred.device), diagonal=1).bool()
                    diff = fc_pred[:, mask] - fc_true[:, mask]
                    loss_fc = (diff.pow(2)).mean()
                    return loss_total + self.fc_loss_weight * loss_fc
                return loss_total
            else:
                raise ValueError(f"Unsupported target shape for training: {tuple(yb.shape)}")

    def train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            self.opt.zero_grad(set_to_none=True)
            loss = self._step(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        self.sched.step()
        return total / max(1, n)

    @torch.no_grad()
    def validate(self, loader) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        for xb, yb in loader:
            loss = self._step(xb, yb)
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        return total / max(1, n)


class TransformerNPI:
    def __init__(self, model: BrainTransformerCausal, perturbation: float = 0.05, device: Optional[str] = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.perturbation = perturbation

    @torch.no_grad()
    def virtual_perturbation(self, windows: torch.Tensor, region_i: int, perturbation: Optional[float] = None) -> torch.Tensor:
        if perturbation is None:
            perturbation = self.perturbation
        windows = windows.to(self.device)
        base_pred = self.model(windows)
        perturbed = windows.clone()
        perturbed[:, -1, region_i] += perturbation
        pert_pred = self.model(perturbed)
        return pert_pred - base_pred

    @torch.no_grad()
    def infer_effective_connectivity(self, windows: torch.Tensor, batch_regions: Optional[int] = None, normalize: bool = True) -> torch.Tensor:
        self.model.eval()
        windows = windows.to(self.device)
        N = windows.size(-1)
        ec = torch.zeros(N, N, device=self.device)
        indices = list(range(N))
        for i in indices:
            resp = self.virtual_perturbation(windows, i)  # [B,N]
            ec[i, :] = resp.mean(dim=0)
        if normalize:
            for i in range(N):
                row = ec[i]
                max_pos = torch.maximum(row.max(), torch.tensor(1e-8, device=row.device))
                if (row > 0).any():
                    scale = torch.maximum(row[row > 0].max(), torch.tensor(1e-8, device=row.device))
                else:
                    scale = row.abs().max().clamp(min=1e-8)
                ec[i] = row / scale
        return ec

    def infer_ec_jacobian(self, window: torch.Tensor, avg_over: int = 64) -> torch.Tensor:
        self.model.eval()
        window = window.to(self.device)
        B, S, N = window.shape
        use = min(B, avg_over)
        J = torch.zeros(N, N, device=self.device)
        for b in range(use):
            xb = window[b:b+1].clone().detach().requires_grad_(True)
            out = self.model(xb)  # [1,N]
            grads = []
            for k in range(N):
                self.model.zero_grad(set_to_none=True)
                g = torch.autograd.grad(out[0, k], xb, retain_graph=True, allow_unused=True)[0]
                if g is None:
                    g = torch.zeros_like(xb)
                grads.append(g[0, -1, :].detach())
            G = torch.stack(grads, dim=0)  # [N,N]
            J += G
        J = (J / float(use))
        return J
