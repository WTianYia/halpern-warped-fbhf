import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MAIN_EXP = ROOT / "experiments" / "02_main_oracle_real_synthetic"
REV_EXP = ROOT / "experiments" / "08_revision_experiments"
sys.path.insert(0, str(MAIN_EXP))
sys.path.insert(0, str(REV_EXP))

from trackB_formal_eval import (  # noqa: E402
    DEV,
    PrecNet,
    TV,
    clamp_ts,
    download_images,
    features,
    make_batch,
    normD,
    rel_error_vec,
    set_seed,
    unroll_final,
    warped_step,
)
from reviewer_experiments import pdhg_final  # noqa: E402


def certified_cfg(args):
    return {
        "tlo": args.tau_min,
        "thi": args.tau_max,
        "slo": args.s_min,
        "shi": args.s_max,
        "rho": args.ell_cap,
        "tau0": args.tau0,
        "s0": args.s0,
        "c": args.c,
    }


def sign_test_p_value(wins, n):
    k = min(wins, n - wins)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def objective_after(prob, xs):
    weights = torch.arange(1, len(xs) + 1, device=xs[0].device, dtype=xs[0].dtype)
    vals = torch.stack([prob.obj(x).mean() for x in xs])
    return (weights * vals).sum() / weights.sum()


def unroll_cert(prob, x0, y0, K, L, cfg, net=None, fixed=None, xstar=None, nx=None):
    x, y = x0, y0
    x_prev = x0
    tau = torch.full_like(x, cfg["tau0"])
    s = torch.full_like(x, cfg["s0"])
    xs = []
    errs = []
    max_tau = 0.0
    max_ell2 = 0.0
    for k in range(K):
        if fixed is not None:
            tau = torch.full_like(x, fixed[0])
            s = torch.full_like(x, fixed[1])
        elif net is not None:
            _, (px0, py0) = warped_step(prob, x, y, torch.full_like(x, cfg["tau0"]), torch.full_like(x, cfg["s0"]))
            raw = net(features(x, y, px0, py0, x_prev))
            dtau, dsig = clamp_ts(raw, L, cfg["tlo"], cfg["thi"], cfg["slo"], cfg["shi"], cfg["rho"])
            eta = cfg["c"] / ((k + 1) ** 1.1)
            tau = torch.clamp(tau + torch.clamp(dtau - tau, -eta, eta), cfg["tlo"], cfg["thi"])
            s = torch.clamp(s + torch.clamp(dsig - s, -eta, eta), cfg["slo"], cfg["shi"])
            s = torch.minimum(s, cfg["rho"] / (tau * L**2 + 1e-9))
        else:
            tau = torch.full_like(x, cfg["tau0"])
            s = torch.full_like(x, cfg["s0"])
        max_tau = max(max_tau, float(tau.max().detach().cpu()))
        max_ell2 = max(max_ell2, float((tau * s * L**2).max().detach().cpu()))
        (x, y), _ = warped_step(prob, x, y, tau, s)
        xs.append(x)
        if xstar is not None and nx is not None:
            errs.append(float(rel_error_vec(x, xstar, nx).mean().detach().cpu()))
        x_prev = x
    if xstar is not None and nx is not None:
        endpoint = rel_error_vec(x, xstar, nx).detach().cpu().numpy()
    else:
        endpoint = None
    return xs, errs, endpoint, {"max_tau": max_tau, "max_ell2": max_ell2}


@torch.no_grad()
def val_objective(net, args, L, cfg, seed):
    b, otf, mu, _ = make_batch(args.batch, args.size, args.size, DEV, seed, "real", "train", "train", args.image_dir)
    prob = TV(b, otf, mu)
    x0 = b.clone()
    y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
    learned = prob.obj(unroll_cert(prob, x0, y0, args.train_K, L, cfg, net=net)[0][-1]).mean().item()
    fixed = prob.obj(unroll_cert(prob, x0, y0, args.train_K, L, cfg, fixed=(args.fixed_tau, args.fixed_s))[0][-1]).mean().item()
    return learned, fixed


def train(args, L, cfg):
    set_seed(args.seed)
    ckpt = Path(args.ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    net = PrecNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), args.lr)
    best = float("inf")
    log_rows = []
    for it in range(1, args.train_iters + 1):
        b, otf, mu, _ = make_batch(
            args.batch,
            args.size,
            args.size,
            DEV,
            args.seed * 100000 + it,
            "real",
            "train",
            "train",
            args.image_dir,
        )
        prob = TV(b, otf, mu)
        x0 = b.clone()
        y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
        xs = unroll_cert(prob, x0, y0, args.train_K, L, cfg, net=net)[0]
        loss = objective_after(prob, xs)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it == 1 or it % args.eval_every == 0:
            net.eval()
            learned, fixed = val_objective(net, args, L, cfg, args.val_seed)
            net.train()
            saved = False
            if math.isfinite(learned) and learned < best:
                best = learned
                torch.save(net.state_dict(), ckpt)
                saved = True
            row = {"iter": it, "loss": float(loss.item()), "val_learned": learned, "val_fixed": fixed, "saved": saved}
            log_rows.append(row)
            print(
                f"it {it:5d} loss {loss.item():.6e} | val learned {learned:.6e} "
                f"vs certified fixed {fixed:.6e}{' *saved' if saved else ''}",
                flush=True,
            )
    return log_rows


@torch.no_grad()
def evaluate(args, L, cfg):
    b, otf, mu, _ = make_batch(args.ntest, args.size, args.size, DEV, args.test_seed, "real", "train", "train", args.image_dir)
    prob = TV(b, otf, mu)
    x0 = b.clone()
    y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
    chi = 4 / (1 + math.sqrt(1 + 16 * L**2))
    old_xstar = unroll_final(prob, x0, y0, args.ref_iters, L=L, cfg=cfg, fixed=(chi - 0.05 * chi, 0.10))
    xstar_half = pdhg_final(prob, x0, y0, max(1, args.pdhg_ref_iters // 2), L=L, tau=args.pdhg_tau)
    xstar = pdhg_final(prob, x0, y0, args.pdhg_ref_iters, L=L, tau=args.pdhg_tau)
    nx = torch.sqrt((xstar**2).sum(dim=(1, 2, 3))).clamp_min(1e-12)

    net = PrecNet().to(DEV)
    net.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    net.eval()

    specs = [
        ("plain-FBHF", None, (chi - 0.05 * chi, 0.10)),
        ("certified-fixed", None, (args.fixed_tau, args.fixed_s)),
        ("certified-learned", net, None),
    ]
    curves = []
    per_image = []
    summary = []
    for method, net_i, fixed in specs:
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, errs, endpoint, diag = unroll_cert(prob, x0, y0, args.eval_K, L, cfg, net=net_i, fixed=fixed, xstar=xstar, nx=nx)
        if DEV == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        endpoint = np.asarray(endpoint, dtype=float)
        oracle = args.eval_K * 4
        for i, err in enumerate(errs, start=1):
            curves.append(
                {
                    "method": method,
                    "iteration": i,
                    "oracle": i * 4,
                    "mean_rel_primal_error": err,
                    "wall_time_s": elapsed * i / args.eval_K,
                }
            )
        for j, val in enumerate(endpoint):
            per_image.append({"method": method, "image_index": j, "rel_primal_error_K": float(val), "oracle_K": oracle})
        summary.append(
            {
                "method": method,
                "err_K_mean": float(endpoint.mean()),
                "err_K_std": float(endpoint.std(ddof=1)),
                "err_K_median": float(np.median(endpoint)),
                "oracle_K": oracle,
                "wall_time_K": elapsed,
                "network_forward_K": args.eval_K if net_i is not None else 0,
                **diag,
            }
        )
    by_method = {}
    for row in per_image:
        by_method.setdefault(row["method"], {})[row["image_index"]] = row["rel_primal_error_K"]
    wins = sum(
        by_method["certified-learned"][j] < by_method["certified-fixed"][j]
        for j in by_method["certified-learned"]
    )
    pdhg_self_gap = rel_error_vec(xstar_half, xstar, nx).detach().cpu().numpy()
    old_ref_gap = rel_error_vec(old_xstar, xstar, nx).detach().cpu().numpy()
    diagnostics = {
        "reference": "Condat-Vu/PDHG",
        "pdhg_ref_iters": int(args.pdhg_ref_iters),
        "pdhg_ref_half_iters": int(max(1, args.pdhg_ref_iters // 2)),
        "old_fbhf_ref_iters_for_gap_only": int(args.ref_iters),
        "ref_self_gap_mean": float(pdhg_self_gap.mean()),
        "ref_self_gap_median": float(np.median(pdhg_self_gap)),
        "ref_self_gap_max": float(pdhg_self_gap.max()),
        "old_fbhf_ref_gap_mean": float(old_ref_gap.mean()),
        "old_fbhf_ref_gap_median": float(np.median(old_ref_gap)),
        "old_fbhf_ref_gap_max": float(old_ref_gap.max()),
        "learned_vs_fixed_wins": int(wins),
        "learned_vs_fixed_sign_p": sign_test_p_value(wins, args.ntest),
    }
    return curves, per_image, summary, diagnostics


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_plot(outdir, curves, summary):
    colors = {
        "plain-FBHF": "#5B6770",
        "certified-fixed": "#3775BA",
        "certified-learned": "#B64342",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=220)
    for method in colors:
        xs = [r["oracle"] for r in curves if r["method"] == method]
        ys = [r["mean_rel_primal_error"] for r in curves if r["method"] == method]
        axes[0].plot(xs, ys, lw=1.7, color=colors[method], label=method)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("dominant oracle calls")
    axes[0].set_ylabel("relative primal error")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].text(-0.13, 1.04, "a", transform=axes[0].transAxes, fontweight="bold")

    labels = [r["method"] for r in summary]
    vals = [r["err_K_mean"] for r in summary]
    stds = [r["err_K_std"] for r in summary]
    axes[1].bar(range(len(labels)), vals, yerr=stds, color=[colors[x] for x in labels], capsize=2, width=0.62)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(["plain", "cert. fixed", "cert. learned"], rotation=25, ha="right")
    axes[1].set_ylabel("endpoint error")
    axes[1].text(-0.13, 1.04, "b", transform=axes[1].transAxes, fontweight="bold")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    fig.tight_layout(w_pad=1.8)
    for ext in ["pdf", "png", "svg", "tiff"]:
        fig.savefig(Path(outdir) / f"fig_certified_mode_supplement.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image_dir", default="standard_images")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--val_seed", type=int, default=9031)
    ap.add_argument("--test_seed", type=int, default=20240731)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ntest", type=int, default=64)
    ap.add_argument("--train_K", type=int, default=40)
    ap.add_argument("--eval_K", type=int, default=1000)
    ap.add_argument("--ref_iters", type=int, default=4000)
    ap.add_argument("--pdhg_ref_iters", type=int, default=100000)
    ap.add_argument("--pdhg_tau", type=float, default=1.0)
    ap.add_argument("--train_iters", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau_min", type=float, default=0.1)
    ap.add_argument("--tau_max", type=float, default=0.6)
    ap.add_argument("--s_min", type=float, default=0.02)
    ap.add_argument("--s_max", type=float, default=0.12)
    ap.add_argument("--ell_cap", type=float, default=0.6)
    ap.add_argument("--tau0", type=float, default=0.3)
    ap.add_argument("--s0", type=float, default=0.1)
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--fixed_tau", type=float, default=0.6)
    ap.add_argument("--fixed_s", type=float, default=0.1)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    download_images(args.image_dir)
    set_seed(args.seed)
    L = normD(args.size, args.size, DEV)
    cfg = certified_cfg(args)
    beta = 1.0
    compatibility_cap = 1.0 - args.tau_max / (2.0 * beta)
    compatibility_margin = compatibility_cap - args.ell_cap
    protocol = {
        "device": DEV,
        "L_D": L,
        "beta": beta,
        "tau_max": args.tau_max,
        "ell_cap": args.ell_cap,
        "compatibility_cap": compatibility_cap,
        "compatibility_margin": compatibility_margin,
        "certified": bool(args.ell_cap < compatibility_cap),
        "fixed_tau": args.fixed_tau,
        "fixed_s": args.fixed_s,
        "fixed_ell2": args.fixed_tau * args.fixed_s * L**2,
        "ntest": args.ntest,
        "eval_K": args.eval_K,
        "ref_iters": args.ref_iters,
        "pdhg_ref_iters": args.pdhg_ref_iters,
    }
    with (outdir / "certified_protocol.json").open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)
    print(json.dumps(protocol, indent=2), flush=True)

    train_log = train(args, L, cfg)
    curves, per_image, summary, diagnostics = evaluate(args, L, cfg)
    write_csv(outdir / "train_log.csv", train_log)
    write_csv(outdir / "curves_certified.csv", curves)
    write_csv(outdir / "per_image_certified.csv", per_image)
    with (outdir / "summary_certified.json").open("w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "summary": summary, "reference_diagnostics": diagnostics}, f, indent=2)
    with (outdir / "reference_diagnostics_certified.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    make_plot(outdir, curves, summary)
    print(json.dumps({"summary": summary, "reference_diagnostics": diagnostics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
