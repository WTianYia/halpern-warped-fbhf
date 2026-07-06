import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def sign_test_p_value(wins, n):
    k = min(wins, n - wins)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def project_simplex(v):
    """Euclidean projection of each row of v onto the probability simplex."""
    u, _ = torch.sort(v, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1) - 1.0
    ind = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype).view(1, -1)
    cond = u - cssv / ind > 0
    rho = cond.sum(dim=1).clamp_min(1)
    theta = cssv.gather(1, (rho - 1).view(-1, 1)) / rho.to(v.dtype).view(-1, 1)
    return torch.clamp(v - theta, min=0.0)


def weighted_project_simplex(v, weights):
    """Projection onto the simplex in the diagonal metric diag(1 / weights)."""
    weights = weights.expand_as(v).clamp_min(1e-12)
    ratio = v / weights
    ratio_sorted, order = torch.sort(ratio, dim=1, descending=True)
    v_sorted = v.gather(1, order)
    w_sorted = weights.gather(1, order)
    prefix_v = torch.cumsum(v_sorted, dim=1)
    prefix_w = torch.cumsum(w_sorted, dim=1).clamp_min(1e-12)
    theta_all = (prefix_v - 1.0) / prefix_w
    active = ratio_sorted > theta_all
    rho = active.sum(dim=1).clamp_min(1)
    theta = theta_all.gather(1, (rho - 1).view(-1, 1))
    return torch.clamp(v - theta * weights, min=0.0)


def huber(r, delta):
    a = torch.abs(r)
    return torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def huber_grad(r, delta):
    return torch.clamp(r, min=-delta, max=delta)


class HuberSaddle:
    def __init__(self, mat, b, delta, lam, rho):
        self.mat = mat
        self.b = b
        self.delta = float(delta)
        self.lam = float(lam)
        self.rho = float(rho)
        self.batch, self.m, self.n = mat.shape
        self.u = torch.full((self.batch, self.m), 1.0 / self.m, device=mat.device, dtype=mat.dtype)
        self._spectral = None

    @staticmethod
    def make(batch, n, m, seed, delta=0.5, lam=0.1, rho=0.2, outlier_low=0.05, outlier_high=0.2, rank_def=0):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        mat = torch.randn(batch, m, n, generator=gen) / math.sqrt(n)
        if rank_def > 0:
            mat[:, :, -rank_def:] = 0.0
        x_true = torch.randn(batch, n, generator=gen) / math.sqrt(n)
        clean = torch.bmm(mat, x_true.unsqueeze(-1)).squeeze(-1)
        b = clean + 0.03 * torch.randn(batch, m, generator=gen)
        probs = outlier_low + (outlier_high - outlier_low) * torch.rand(batch, generator=gen)
        mask = torch.rand(batch, m, generator=gen) < probs.view(batch, 1)
        signs = torch.where(torch.rand(batch, m, generator=gen) < 0.5, -1.0, 1.0)
        amp = 4.0 + 4.0 * torch.rand(batch, m, generator=gen)
        b = b + mask.to(b.dtype) * signs * amp
        return HuberSaddle(mat.to(DEV), b.to(DEV), delta, lam, rho), x_true.to(DEV)

    def residuals(self, x):
        return torch.bmm(self.mat, x.unsqueeze(-1)).squeeze(-1) - self.b

    def h_and_grad(self, x):
        r = self.residuals(x)
        return huber(r, self.delta), huber_grad(r, self.delta)

    def B(self, x, y):
        h, psi = self.h_and_grad(x)
        bx = torch.bmm(self.mat.transpose(1, 2), (y * psi).unsqueeze(-1)).squeeze(-1)
        by = -h
        return bx, by

    def C(self, x, y):
        return self.lam * x, self.rho * (y - self.u)

    def G(self, x, y):
        bx, by = self.B(x, y)
        cx, cy = self.C(x, y)
        return bx + cx, by + cy

    def merit(self, x, y):
        gx, gy = self.G(x, y)
        # The y part uses projected gradient residual for the simplex block.
        py = project_simplex(y - gy)
        return torch.sqrt((gx * gx).sum(dim=1) + ((y - py) ** 2).sum(dim=1))

    def spectral_norm(self, iters=40):
        if self._spectral is not None:
            return self._spectral
        v = torch.randn(self.batch, self.n, device=self.mat.device)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        for _ in range(iters):
            av = torch.bmm(self.mat, v.unsqueeze(-1)).squeeze(-1)
            v = torch.bmm(self.mat.transpose(1, 2), av.unsqueeze(-1)).squeeze(-1)
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        av = torch.bmm(self.mat, v.unsqueeze(-1)).squeeze(-1)
        self._spectral = av.norm(dim=1)
        return self._spectral

    def LB_bound(self):
        spec = self.spectral_norm()
        return spec * spec / self.delta + self.delta * spec

    def beta(self):
        mc = max(self.lam, self.rho)
        return float("inf") if mc <= 0 else 1.0 / mc

    def chi(self):
        beta = self.beta()
        lb = float(self.LB_bound().max().item())
        if math.isinf(beta):
            return 1.0 / max(lb, 1e-12)
        return 4.0 * beta / (1.0 + math.sqrt(1.0 + 16.0 * beta * beta * lb * lb))

    def LG_bound(self):
        return self.LB_bound() + max(self.lam, self.rho)


class MetricNet(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.x_net = nn.Sequential(nn.Linear(5, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.s_net = nn.Sequential(nn.Linear(8, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.x_net[-1].weight)
        nn.init.zeros_(self.x_net[-1].bias)
        nn.init.zeros_(self.s_net[-1].weight)
        nn.init.zeros_(self.s_net[-1].bias)

    def forward(self, prob, x, y, x_prev, y_prev, gx, gy):
        x_feat = torch.stack(
            [
                x,
                gx,
                x - x_prev,
                torch.abs(gx),
                torch.full_like(x, float(prob.lam)),
            ],
            dim=-1,
        )
        raw_tau = self.x_net(x_feat).squeeze(-1)
        h, psi = prob.h_and_grad(x)
        h_mean = h.mean(dim=1, keepdim=True)
        h_std = h.std(dim=1, keepdim=True).clamp_min(1e-12)
        psi_abs = psi.abs()
        psi_mean = psi_abs.mean(dim=1, keepdim=True)
        psi_std = psi_abs.std(dim=1, keepdim=True).clamp_min(1e-12)
        y_mean = y.mean(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True).clamp_min(1e-12)
        s_feat = torch.stack(
            [
                h,
                (h - h_mean) / h_std,
                psi_abs,
                (psi_abs - psi_mean) / psi_std,
                y,
                (y - y_mean) / y_std,
                gy,
                y - y_prev,
            ],
            dim=-1,
        )
        raw_s = self.s_net(s_feat).squeeze(-1)
        return raw_tau, raw_s


def metric_bounds(cfg, chi):
    scale = float(chi) if cfg.get("scale_by_chi", True) else 1.0
    return cfg["tlo"] * scale, cfg["thi"] * scale, cfg["slo"] * scale, cfg["shi"] * scale


def clamp_metric(raw_tau, raw_s, cfg, chi):
    lo, hi, s_lo, s_hi = metric_bounds(cfg, chi)
    anchor_tau_mult = cfg.get("anchor_tau_mult", 0.0)
    anchor_s_mult = cfg.get("anchor_s_mult", 0.0)
    if anchor_tau_mult > 0.0:
        anchor_tau = anchor_tau_mult * chi
        delta_tau = cfg.get("anchor_delta_mult", 0.15) * chi
        tau = torch.clamp(anchor_tau + delta_tau * torch.tanh(raw_tau), min=lo, max=hi)
    else:
        tau = lo + (hi - lo) * torch.sigmoid(raw_tau)
    if anchor_s_mult > 0.0:
        anchor_s = anchor_s_mult * chi
        delta_s = cfg.get("anchor_delta_mult", 0.15) * chi
        s = torch.clamp(anchor_s + delta_s * torch.tanh(raw_s), min=s_lo, max=s_hi)
    else:
        s = s_lo + (s_hi - s_lo) * torch.sigmoid(raw_s)
    return tau, s


def bv_limit(new, old, radius):
    diff = torch.clamp(new - old, min=-radius, max=radius)
    return old + diff


def fbhf_step(prob, x, y, tau, s, return_test=False):
    gx, gy = prob.G(x, y)
    px = x - tau * gx
    py = weighted_project_simplex(y - s * gy, s)
    bx, by = prob.B(x, y)
    bpx, bpy = prob.B(px, py)
    tx = px + tau * (bx - bpx)
    ty = py + s * (by - bpy)
    ty = weighted_project_simplex(ty, s)
    if not return_test:
        return tx, ty, px, py
    dx = x - px
    dy = y - py
    zp_m = ((dx * dx) / tau).sum(dim=1) + ((dy * dy) / s).sum(dim=1)
    dbx = bx - bpx
    dby = by - bpy
    b_m_inv = (tau * dbx * dbx).sum(dim=1) + (s * dby * dby).sum(dim=1)
    ell2 = b_m_inv / zp_m.clamp_min(1e-30)
    if math.isinf(prob.beta()):
        compat_cap = torch.ones_like(ell2)
    else:
        kappa = torch.maximum(tau.max(dim=1).values, s.max(dim=1).values)
        compat_cap = 1.0 - kappa / (2.0 * prob.beta())
    test = {"ell2": ell2, "compat_cap": compat_cap, "zp_m": zp_m}
    return tx, ty, px, py, test


def tested_fbhf_step(prob, x, y, tau, s, chi, cfg):
    shrink = cfg.get("test_shrink", 0.7)
    max_backtracks = int(cfg.get("test_max_backtracks", 12))
    safety = cfg.get("test_safety", 0.98)
    fallback_tau = cfg.get("test_fallback_mult", 0.9) * chi
    fallback_s = cfg.get("test_fallback_mult", 0.9) * chi
    backtracks = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
    tau_try = tau
    s_try = s
    accepted = None
    for _ in range(max_backtracks + 1):
        tx, ty, px, py, test = fbhf_step(prob, x, y, tau_try, s_try, return_test=True)
        threshold = safety * test["compat_cap"]
        passed = (test["zp_m"] <= 1e-24) | ((threshold > 0.0) & (test["ell2"] <= threshold))
        accepted = (tx, ty, px, py, test, passed)
        if bool(passed.all().item()):
            break
        fail = (~passed).view(-1, 1)
        tau_try = torch.where(fail, tau_try * shrink, tau_try)
        s_try = torch.where(fail, s_try * shrink, s_try)
        backtracks = backtracks + (~passed).to(torch.long)
    tx, ty, px, py, test, passed = accepted
    if not bool(passed.all().item()):
        fail = (~passed).view(-1, 1)
        tau_try = torch.where(fail, torch.full_like(tau_try, fallback_tau), tau_try)
        s_try = torch.where(fail, torch.full_like(s_try, fallback_s), s_try)
        tx, ty, px, py, test = fbhf_step(prob, x, y, tau_try, s_try, return_test=True)
        passed = torch.ones_like(test["ell2"], dtype=torch.bool)
    stats = {
        "backtracks": backtracks,
        "ell2": test["ell2"].detach(),
        "compat_cap": test["compat_cap"].detach(),
        "tau": tau_try.detach(),
        "s": s_try.detach(),
    }
    return tx, ty, px, py, stats


def update_runtime_stats(target, stats):
    if target is None or stats is None:
        return
    target["steps"] += int(stats["backtracks"].numel())
    target["backtracks"] += int(stats["backtracks"].sum().item())
    target["max_backtracks"] = max(target["max_backtracks"], int(stats["backtracks"].max().item()))
    target["max_ell2"] = max(target["max_ell2"], float(stats["ell2"].max().item()))
    target["min_margin"] = min(target["min_margin"], float((stats["compat_cap"] - stats["ell2"]).min().item()))
    target["max_tau_over_chi"] = max(target["max_tau_over_chi"], float((stats["tau"] / target["chi"]).max().item()))
    target["max_s_over_chi"] = max(target["max_s_over_chi"], float((stats["s"] / target["chi"]).max().item()))


def fbf_step(prob, x, y, gamma):
    gx, gy = prob.G(x, y)
    px = x - gamma * gx
    py = project_simplex(y - gamma * gy)
    gpx, gpy = prob.G(px, py)
    tx = px + gamma * (gx - gpx)
    ty = py + gamma * (gy - gpy)
    return tx, project_simplex(ty)


def eg_step(prob, x, y, gamma):
    gx, gy = prob.G(x, y)
    px = x - gamma * gx
    py = project_simplex(y - gamma * gy)
    gpx, gpy = prob.G(px, py)
    return x - gamma * gpx, project_simplex(y - gamma * gpy)


def frb_step(prob, x, y, gx_prev, gy_prev, gamma):
    gx, gy = prob.G(x, y)
    tx = x - gamma * (2.0 * gx - gx_prev)
    ty = project_simplex(y - gamma * (2.0 * gy - gy_prev))
    return tx, ty, gx, gy


def run_trace(prob, method, steps, net=None, fixed=None, cfg=None, checkpoints=None, anchor=False, runtime_stats=None):
    if checkpoints is None:
        checkpoints = []
    checkpoints = set(int(c) for c in checkpoints)
    x = torch.zeros(prob.batch, prob.n, device=prob.mat.device)
    y = prob.u.clone()
    x_prev = x.clone()
    y_prev = y.clone()
    tau_prev = None
    s_prev = None
    gx_prev, gy_prev = prob.G(x, y)
    out = {}
    if 0 in checkpoints:
        out[0] = (x.clone(), y.clone())
    chi = prob.chi()
    lg = float(prob.LG_bound().max().item())
    gamma_fbf = 0.72 / max(lg, 1e-12)
    gamma_frb = 0.45 / max(lg, 1e-12)
    for k in range(1, steps + 1):
        if method in {"plain", "fixed", "learned"}:
            if method == "plain":
                tau = torch.full_like(x, 0.9 * chi)
                s = torch.full((prob.batch, 1), 0.9 * chi, device=x.device, dtype=x.dtype)
            elif method == "fixed":
                tau_v, s_v = fixed
                tau = torch.full_like(x, tau_v)
                s = torch.full((prob.batch, 1), s_v, device=x.device, dtype=x.dtype)
            else:
                gx, gy = prob.G(x, y)
                raw_tau, raw_s = net(prob, x, y, x_prev, y_prev, gx, gy)
                tau, s = clamp_metric(raw_tau, raw_s, cfg, chi)
                if cfg.get("bv", True):
                    radius = cfg["c"] / ((k + 1.0) ** cfg.get("p", 1.1))
                    if tau_prev is not None:
                        tau = bv_limit(tau, tau_prev, radius)
                        s = bv_limit(s, s_prev, radius)
                    tau_prev = tau.detach()
                    s_prev = s.detach()
            if cfg is not None and cfg.get("runtime_test", False):
                tx, ty, px, py, step_stats = tested_fbhf_step(prob, x, y, tau, s, chi, cfg)
                update_runtime_stats(runtime_stats, step_stats)
            else:
                tx, ty, px, py = fbhf_step(prob, x, y, tau, s)
            if anchor:
                lam = 1.0 / (k + 2.0)
                tx = (1 - lam) * tx
                ty = project_simplex((1 - lam) * ty + lam * prob.u)
            x_prev, y_prev = x, y
            x, y = tx, ty
        elif method == "fbf":
            x, y = fbf_step(prob, x, y, gamma_fbf)
        elif method == "eg":
            x, y = eg_step(prob, x, y, gamma_fbf)
        elif method == "frb":
            x, y, gx_prev, gy_prev = frb_step(prob, x, y, gx_prev, gy_prev, gamma_frb)
        else:
            raise ValueError(method)
        if k in checkpoints:
            out[k] = (x.clone(), y.clone())
    return x, y, out


def rel_error(x, y, xref, yref):
    denom = torch.sqrt((xref * xref).sum(dim=1) + (yref * yref).sum(dim=1)).clamp_min(1e-12)
    return torch.sqrt(((x - xref) ** 2).sum(dim=1) + ((y - yref) ** 2).sum(dim=1)) / denom


def train_metric(args, outdir):
    set_seed(args.seed)
    outdir.mkdir(parents=True, exist_ok=True)
    net = MetricNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    cfg = dict(
        tlo=args.tlo,
        thi=args.thi,
        slo=args.slo,
        shi=args.shi,
        c=args.bv_c,
        p=1.1,
        bv=True,
        scale_by_chi=True,
        anchor_tau_mult=args.learn_anchor_tau_mult,
        anchor_s_mult=args.learn_anchor_s_mult,
        anchor_delta_mult=args.learn_anchor_delta_mult,
        runtime_test=args.runtime_test,
        test_safety=args.test_safety,
        test_shrink=args.test_shrink,
        test_max_backtracks=args.test_max_backtracks,
        test_fallback_mult=args.test_fallback_mult,
    )
    best = float("inf")
    rows = []
    for it in range(1, args.train_iters + 1):
        prob, _ = HuberSaddle.make(
            args.batch,
            args.n,
            args.m,
            args.seed * 100000 + it,
            args.delta,
            args.lam,
            args.rho,
            rank_def=0,
        )
        x, y, _ = run_trace(prob, "learned", args.train_K, net=net, cfg=cfg)
        if args.train_loss == "reference":
            with torch.no_grad():
                xr, yr, _ = run_trace(prob, "fbf", args.train_ref_steps)
            loss = rel_error(x, y, xr, yr).mean()
        else:
            loss = prob.merit(x, y).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it == 1 or it % args.eval_every == 0:
            with torch.no_grad():
                val_prob, _ = HuberSaddle.make(args.batch, args.n, args.m, args.val_seed + it, args.delta, args.lam, args.rho)
                xv, yv, _ = run_trace(val_prob, "learned", args.train_K, net=net, cfg=cfg)
                val_merit = val_prob.merit(xv, yv).mean().item()
                if args.train_loss == "reference":
                    xrv, yrv, _ = run_trace(val_prob, "fbf", args.train_ref_steps)
                    val_score = rel_error(xv, yv, xrv, yrv).mean().item()
                else:
                    val_score = val_merit
                row = {
                    "iter": it,
                    "train_score": float(loss.item()),
                    "val_score": val_score,
                    "val_merit": val_merit,
                }
                rows.append(row)
                if math.isfinite(val_score) and val_score < best:
                    best = val_score
                    torch.save(net.state_dict(), outdir / "nonlinear_saddle_metric.pt")
                    row["saved"] = 1
                else:
                    row["saved"] = 0
                print(json.dumps(row), flush=True)
    write_csv(outdir / "train_log.csv", rows)
    return outdir / "nonlinear_saddle_metric.pt"


@torch.no_grad()
def tune_fixed(prob, candidates, eval_steps):
    best = None
    rows = []
    for tau, s in candidates:
        x, y, _ = run_trace(prob, "fixed", eval_steps, fixed=(tau, s))
        merit = prob.merit(x, y).mean().item()
        row = {"tau": tau, "s": s, "validation_mean_merit": merit}
        rows.append(row)
        if math.isfinite(merit) and (best is None or merit < best[0]):
            best = (merit, tau, s)
    return best, rows


@torch.no_grad()
def evaluate_main(args, outdir, ckpt=None):
    outdir.mkdir(parents=True, exist_ok=True)
    prob, _ = HuberSaddle.make(args.ntest, args.n, args.m, args.eval_seed, args.delta, args.lam, args.rho)
    chi = prob.chi()
    fixed_grid = []
    for mt in args.fixed_tau_mult:
        for ms in args.fixed_s_mult:
            fixed_grid.append((float(mt * chi), float(ms * chi)))
    val_prob, _ = HuberSaddle.make(args.ntest, args.n, args.m, args.val_seed + int(1000 * args.rho), args.delta, args.lam, args.rho)
    best, fixed_rows = tune_fixed(val_prob, fixed_grid, args.eval_steps)
    fixed_tau, fixed_s = best[1], best[2]
    write_csv(outdir / "fixed_grid.csv", fixed_rows)
    ref_x, ref_y, saved = run_trace(prob, "fbf", args.ref_steps, checkpoints=[args.ref_steps // 2, args.ref_steps])
    half_x, half_y = saved[args.ref_steps // 2]
    ref_gap = rel_error(half_x, half_y, ref_x, ref_y)
    cfg = dict(
        tlo=args.tlo,
        thi=args.thi,
        slo=args.slo,
        shi=args.shi,
        c=args.bv_c,
        p=1.1,
        bv=True,
        scale_by_chi=True,
        anchor_tau_mult=args.learn_anchor_tau_mult,
        anchor_s_mult=args.learn_anchor_s_mult,
        anchor_delta_mult=args.learn_anchor_delta_mult,
        runtime_test=args.runtime_test,
        test_safety=args.test_safety,
        test_shrink=args.test_shrink,
        test_max_backtracks=args.test_max_backtracks,
        test_fallback_mult=args.test_fallback_mult,
    )
    net = None
    if ckpt is not None and Path(ckpt).exists():
        net = MetricNet().to(DEV)
        net.load_state_dict(torch.load(ckpt, map_location=DEV))
        net.eval()
        x0 = torch.zeros(prob.batch, prob.n, device=prob.mat.device)
        y0 = prob.u.clone()
        gx0, gy0 = prob.G(x0, y0)
        raw_tau0, raw_s0 = net(prob, x0, y0, x0, y0, gx0, gy0)
        tau0, s0 = clamp_metric(raw_tau0, raw_s0, cfg, chi)
    methods = [
        ("plain-FBHF", "plain", None),
        ("fixed-warped-FBHF", "fixed", (fixed_tau, fixed_s)),
        ("Tseng-FBF", "fbf", None),
        ("extragradient", "eg", None),
        ("FRB", "frb", None),
    ]
    if net is not None:
        methods.append(("learned-warped-FBHF", "learned", None))
    checkpoints = sorted(set([0, args.eval_steps // 4, args.eval_steps // 2, args.eval_steps]))
    curve_rows = []
    per_rows = []
    summary = []
    traces = {}
    runtime_stats_by_method = {}
    for label, method, fixed in methods:
        runtime_stats = None
        if cfg.get("runtime_test", False) and method in {"plain", "fixed", "learned"}:
            runtime_stats = {
                "chi": float(chi),
                "steps": 0,
                "backtracks": 0,
                "max_backtracks": 0,
                "max_ell2": 0.0,
                "min_margin": float("inf"),
                "max_tau_over_chi": 0.0,
                "max_s_over_chi": 0.0,
            }
        t0 = time.perf_counter()
        x, y, trace = run_trace(prob, method, args.eval_steps, net=net, fixed=fixed, cfg=cfg, checkpoints=checkpoints, runtime_stats=runtime_stats)
        wall = time.perf_counter() - t0
        if runtime_stats is not None:
            runtime_stats_by_method[label] = runtime_stats
        charged_calls = float(dominant_calls(method, args.eval_steps))
        if runtime_stats is not None:
            charged_calls += 2.0 * runtime_stats["backtracks"] / max(args.ntest, 1)
        err = rel_error(x, y, ref_x, ref_y).detach().cpu().numpy()
        merit = prob.merit(x, y).detach().cpu().numpy()
        for i in range(args.ntest):
            per_rows.append({"method": label, "instance": i, "rel_error": float(err[i]), "merit": float(merit[i])})
        for k, (tx, ty) in trace.items():
            e = rel_error(tx, ty, ref_x, ref_y).mean().item()
            curve_rows.append({"method": label, "iter": k, "dominant_B_calls": dominant_calls(method, k), "mean_rel_error": e})
        summary.append(
            {
                "method": label,
                "mean_rel_error": float(err.mean()),
                "std_rel_error": float(err.std(ddof=0)),
                "median_rel_error": float(np.median(err)),
                "mean_merit": float(merit.mean()),
                "wall_time": float(wall),
                "dominant_B_calls": charged_calls,
            }
        )
        traces[label] = err
    if "learned-warped-FBHF" in traces:
        diff = traces["learned-warped-FBHF"] - traces["fixed-warped-FBHF"]
        tol = 1e-12
        wins = int((diff < -tol).sum())
        losses = int((diff > tol).sum())
        ties = int(args.ntest - wins - losses)
        pval = 1.0 if wins + losses == 0 else sign_test_p_value(wins, wins + losses)
    else:
        wins, losses, ties, pval = None, None, None, None
    diag = {
        "reference": "Tseng-FBF",
        "ref_steps": args.ref_steps,
        "ref_self_gap_mean": float(ref_gap.mean().item()),
        "ref_self_gap_median": float(ref_gap.median().item()),
        "ref_self_gap_max": float(ref_gap.max().item()),
        "chi": float(chi),
        "LB_bound_max": float(prob.LB_bound().max().item()),
        "LG_bound_max": float(prob.LG_bound().max().item()),
        "fixed_tau": float(fixed_tau),
        "fixed_s": float(fixed_s),
        "learned_tau_over_chi_mean_initial": None if net is None else float((tau0 / chi).mean().item()),
        "learned_tau_over_chi_max_initial": None if net is None else float((tau0 / chi).max().item()),
        "learned_s_over_chi_mean_initial": None if net is None else float((s0 / chi).mean().item()),
        "learned_s_over_chi_max_initial": None if net is None else float((s0 / chi).max().item()),
        "learned_vs_fixed_wins": wins,
        "learned_vs_fixed_losses": losses,
        "learned_vs_fixed_ties": ties,
        "learned_vs_fixed_sign_p": pval,
        "runtime_test_enabled": bool(cfg.get("runtime_test", False)),
        "runtime_test_stats": runtime_stats_by_method,
    }
    write_csv(outdir / "main_summary.csv", summary)
    write_csv(outdir / "main_per_instance.csv", per_rows)
    write_csv(outdir / "main_curves.csv", curve_rows)
    with (outdir / "main_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    make_main_plot(outdir, curve_rows, summary)
    return summary, diag


def dominant_calls(method, k):
    if method in {"plain", "fixed", "learned"} or "FBHF" in method:
        return 2 * k
    if method in {"fbf", "eg"} or method in {"Tseng-FBF", "extragradient"}:
        return 2 * k
    if method == "frb" or method == "FRB":
        return k + 1
    return k


@torch.no_grad()
def beta_scan(args, outdir, ckpt=None):
    rows = []
    ckpt_path = Path(ckpt) if ckpt else None
    for rho in args.scan_rho:
        local = argparse.Namespace(**vars(args))
        local.rho = float(rho)
        scan_dir = outdir / f"rho_{rho:g}"
        summary, diag = evaluate_main(local, scan_dir, ckpt_path)
        by = {r["method"]: r for r in summary}
        fixed = by["fixed-warped-FBHF"]["mean_rel_error"]
        for method, r in by.items():
            rows.append(
                {
                    "rho": float(rho),
                    "beta_LB": float((1.0 / max(local.lam, local.rho)) * diag["LB_bound_max"]),
                    "method": method,
                    "mean_rel_error": r["mean_rel_error"],
                    "gain_vs_fixed": float(1.0 - r["mean_rel_error"] / fixed) if fixed > 0 else "",
                }
            )
    write_csv(outdir / "beta_scan_summary.csv", rows)
    make_beta_plot(outdir, rows)


@torch.no_grad()
def evaluate_selection(args, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    rank_def = max(1, args.rank_def)
    prob, _ = HuberSaddle.make(args.ntest, args.n, args.m, args.selection_seed, args.delta, 0.0, args.rho, rank_def=rank_def)
    ref_x, ref_y, _ = run_trace(prob, "plain", args.ref_steps)
    # Since the last rank_def coordinates are invisible to A and lambda=0, the
    # minimum-norm representative has zeros in those coordinates.
    min_x = ref_x.clone()
    min_x[:, -rank_def:] = 0.0
    min_y = ref_y
    starts = [-3.0, 0.0, 3.0]
    rows = []
    for start in starts:
        x0_tail = torch.zeros(prob.batch, prob.n, device=DEV)
        x0_tail[:, -rank_def:] = start
        # Run custom initialisation by shifting the invisible component after
        # every first state; plain keeps it, Halpern damps it to the anchor.
        x_plain, y_plain = x0_tail.clone(), prob.u.clone()
        x_hal, y_hal = x0_tail.clone(), prob.u.clone()
        chi = prob.chi()
        for k in range(1, args.eval_steps + 1):
            x_plain, y_plain, _, _ = fbhf_step(prob, x_plain, y_plain, torch.full_like(x_plain, 0.9 * chi), torch.full((prob.batch, 1), 0.9 * chi, device=DEV))
            tx, ty, _, _ = fbhf_step(prob, x_hal, y_hal, torch.full_like(x_hal, 0.9 * chi), torch.full((prob.batch, 1), 0.9 * chi, device=DEV))
            lam = 1.0 / (k + 2.0)
            x_hal = (1 - lam) * tx
            y_hal = project_simplex((1 - lam) * ty + lam * prob.u)
        for label, x, y in [("plain-FBHF", x_plain, y_plain), ("Halpern-FBHF", x_hal, y_hal)]:
            dist = rel_error(x, y, min_x, min_y).detach().cpu().numpy()
            tail = x[:, -rank_def:].norm(dim=1).detach().cpu().numpy()
            rows.append({"start_tail": start, "method": label, "mean_dist_to_min_norm": float(dist.mean()), "mean_tail_norm": float(tail.mean())})
    write_csv(outdir / "selection_summary.csv", rows)
    make_selection_plot(outdir, rows)
    return rows


def make_main_plot(outdir, curves, summary):
    outdir = Path(outdir)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for method in sorted(set(r["method"] for r in curves)):
        xs = [r["dominant_B_calls"] for r in curves if r["method"] == method]
        ys = [r["mean_rel_error"] for r in curves if r["method"] == method]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=method)
    ax.set_yscale("log")
    ax.set_xlabel("dominant B evaluations")
    ax.set_ylabel("relative error to high-precision reference")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(outdir / f"fig_nonlinear_main.{ext}", dpi=300)
    plt.close(fig)


def make_beta_plot(outdir, rows):
    outdir = Path(outdir)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    methods = [m for m in sorted(set(r["method"] for r in rows)) if m != "fixed-warped-FBHF"]
    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        xs = [r["beta_LB"] for r in sub]
        ys = [r["gain_vs_fixed"] for r in sub]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=method)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta L_B$ estimate")
    ax.set_ylabel("gain vs fixed warped FBHF")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(outdir / f"fig_beta_scan.{ext}", dpi=300)
    plt.close(fig)


def make_selection_plot(outdir, rows):
    outdir = Path(outdir)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    labels = sorted(set(r["method"] for r in rows))
    starts = sorted(set(r["start_tail"] for r in rows))
    width = 0.35
    x = np.arange(len(starts))
    for j, label in enumerate(labels):
        vals = [next(r["mean_tail_norm"] for r in rows if r["method"] == label and r["start_tail"] == s) for s in starts]
        ax.bar(x + (j - 0.5) * width, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in starts])
    ax.set_xlabel("initial invisible component")
    ax.set_ylabel("tail norm after fixed budget")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(outdir / f"fig_selection_nonlinear.{ext}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="nonlinear_run")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--val_seed", type=int, default=7001)
    ap.add_argument("--eval_seed", type=int, default=9001)
    ap.add_argument("--selection_seed", type=int, default=9101)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ntest", type=int, default=64)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--rank_def", type=int, default=20)
    ap.add_argument("--train_K", type=int, default=40)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--ref_steps", type=int, default=100000)
    ap.add_argument("--train_iters", type=int, default=3000)
    ap.add_argument("--train_loss", choices=["merit", "reference"], default="merit")
    ap.add_argument("--train_ref_steps", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tlo", type=float, default=0.25)
    ap.add_argument("--thi", type=float, default=2.5)
    ap.add_argument("--slo", type=float, default=0.25)
    ap.add_argument("--shi", type=float, default=2.5)
    ap.add_argument("--bv_c", type=float, default=0.5)
    ap.add_argument("--learn_anchor_tau_mult", type=float, default=0.0)
    ap.add_argument("--learn_anchor_s_mult", type=float, default=0.0)
    ap.add_argument("--learn_anchor_delta_mult", type=float, default=0.15)
    ap.add_argument("--runtime_test", action="store_true")
    ap.add_argument("--test_safety", type=float, default=0.98)
    ap.add_argument("--test_shrink", type=float, default=0.7)
    ap.add_argument("--test_max_backtracks", type=int, default=12)
    ap.add_argument("--test_fallback_mult", type=float, default=0.9)
    ap.add_argument("--fixed_tau_mult", nargs="+", type=float, default=[0.5, 0.9, 1.3, 1.7, 2.1, 2.5])
    ap.add_argument("--fixed_s_mult", nargs="+", type=float, default=[0.5, 0.9, 1.3, 1.7, 2.1, 2.5])
    ap.add_argument("--scan_rho", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_scan", action="store_true")
    ap.add_argument("--skip_selection", action="store_true")
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"device": DEV}, f, indent=2)
    ckpt = Path(args.ckpt) if args.ckpt else outdir / "nonlinear_saddle_metric.pt"
    if not args.skip_train and not ckpt.exists():
        ckpt = train_metric(args, outdir)
    evaluate_main(args, outdir / "main", ckpt if ckpt.exists() else None)
    if not args.skip_scan:
        beta_scan(args, outdir / "beta_scan", ckpt if ckpt.exists() else None)
    if not args.skip_selection:
        evaluate_selection(args, outdir / "selection")


if __name__ == "__main__":
    main()
