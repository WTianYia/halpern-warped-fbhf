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
CORE_DIR = ROOT / "experiments" / "02_main_oracle_real_synthetic"
sys.path.insert(0, str(CORE_DIR))

import trackB_formal_eval as core  # noqa: E402

DEV = core.DEV


def build_cfg(c, L):
    return {
        "tlo": 0.1,
        "thi": 1.8,
        "slo": 0.02,
        "shi": 0.12,
        "rho": 0.9,
        "tau0": 0.3,
        "s0": 0.1,
        "c": c,
    }


@torch.no_grad()
def pdhg_step(prob, x, y, xbar, tau, sigma, theta):
    y_new = core.projb(y + sigma * core.Dop(xbar), prob.mu)
    grad = core.Ktf(core.Kf(x, prob.otf) - prob.b, prob.otf)
    x_new = x - tau * (grad + core.Dtop(y_new))
    xbar_new = x_new + theta * (x_new - x)
    return x_new, y_new, xbar_new


@torch.no_grad()
def pdhg_reference(prob, x0, y0, L, steps, checkpoints, tau=1.0, theta=1.0):
    sigma = 0.99 / (tau * L**2)
    x = x0.clone()
    y = y0.clone()
    xbar = x0.clone()
    checkpoints = set(checkpoints)
    saved = {}
    prev_x = x.clone()
    last_rel_step = None
    for k in range(1, steps + 1):
        prev_x = x
        x, y, xbar = pdhg_step(prob, x, y, xbar, tau, sigma, theta)
        if k in checkpoints:
            saved[k] = x.clone()
        if k == steps:
            denom = torch.sqrt((x**2).sum(dim=(1, 2, 3))).clamp_min(1e-12)
            last_rel_step = core.rel_error_vec(x, prev_x, denom).detach().cpu().numpy()
    return x, saved, last_rel_step


@torch.no_grad()
def pdhg_trace(prob, x0, y0, L, steps, xstar, nx, tau=1.0, theta=1.0):
    sigma = 0.99 / (tau * L**2)
    x = x0.clone()
    y = y0.clone()
    xbar = x0.clone()
    errs = []
    for _ in range(steps):
        x, y, xbar = pdhg_step(prob, x, y, xbar, tau, sigma, theta)
        errs.append(core.rel_error(x, xstar, nx))
    endpoint = core.rel_error_vec(x, xstar, nx).detach().cpu().numpy()
    return errs, endpoint


@torch.no_grad()
def run_fbhf_method(prob, x0, y0, L, cfg, steps, xstar, nx, method, net=None, chi=None):
    if method == "plain-FBHF":
        counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
        errs, endpoint = core.unroll_trace(
            prob,
            x0,
            y0,
            steps,
            xstar,
            nx,
            net=None,
            mode="bv",
            L=L,
            cfg=cfg,
            fixed=(chi - 0.05 * chi, 0.10),
            counters=counters,
        )
        oracle = (np.arange(1, len(errs) + 1) * 4).tolist()
        return errs, np.asarray(endpoint, dtype=float), oracle, counters
    if method == "line-search-FBHF":
        errs, endpoint, oracle, counters = core.unroll_linesearch_trace(
            prob, x0, y0, steps, chi=chi, xstar=xstar, nx=nx
        )
        return errs, np.asarray(endpoint, dtype=float), oracle, counters
    if method == "fixed-precond":
        counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
        errs, endpoint = core.unroll_trace(
            prob,
            x0,
            y0,
            steps,
            xstar,
            nx,
            net=None,
            mode="bv",
            L=L,
            cfg=cfg,
            fixed=(1.5, 0.07),
            counters=counters,
        )
        oracle = (np.arange(1, len(errs) + 1) * 4).tolist()
        return errs, np.asarray(endpoint, dtype=float), oracle, counters
    if method == "learned-warped-bv":
        counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
        errs, endpoint = core.unroll_trace(
            prob,
            x0,
            y0,
            steps,
            xstar,
            nx,
            net=net,
            mode="bv",
            L=L,
            cfg=cfg,
            fixed=None,
            counters=counters,
        )
        oracle = (np.arange(1, len(errs) + 1) * 4).tolist()
        return errs, np.asarray(endpoint, dtype=float), oracle, counters
    raise ValueError(method)


def endpoint_summary(endpoint):
    arr = np.asarray(endpoint, dtype=float)
    return {
        "err_K_mean": float(arr.mean()),
        "err_K_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "err_K_median": float(np.median(arr)),
    }


def sign_test_p_value(wins, n):
    # two-sided exact binomial test under p=1/2
    k = min(wins, n - wins)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_ref_gap(outdir, rows):
    by_data = sorted({r["data"] for r in rows})
    fig, axes = plt.subplots(1, len(by_data), figsize=(5.8 * len(by_data), 4.2), squeeze=False)
    for ax, data in zip(axes[0], by_data):
        sub = [r for r in rows if r["data"] == data]
        x = np.asarray([r["ref_self_gap_50k_100k"] for r in sub], dtype=float)
        y = np.asarray([r["fixed_minus_learned_absdiff"] for r in sub], dtype=float)
        old = np.asarray([r["old_fbhf4000_gap_to_ref"] for r in sub], dtype=float)
        ax.scatter(x, y, s=28, alpha=0.82, label="50k--100k reference gap")
        ax.scatter(old, y, s=18, alpha=0.45, label="old 4k FBHF reference gap")
        lo = min(x.min(), old.min(), y.min()) * 0.8
        hi = max(x.max(), old.max(), y.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], color="#777777", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("reference gap")
        ax.set_ylabel("|fixed error - learned error|")
        ax.set_title(f"{data}: reference gap vs method difference")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(outdir / f"fig_high_precision_ref_gap.{ext}", dpi=300)
    plt.close(fig)


@torch.no_grad()
def evaluate_condition(args, data, blur, noise):
    core.set_seed(args.seed)
    H = W = args.size
    L = core.normD(H, W, DEV)
    cfg = build_cfg(args.c, L)
    b, otf, mu, _ = core.make_batch(args.ntest, H, W, DEV, args.test_seed, data, blur, noise, args.image_dir)
    prob = core.TV(b, otf, mu)
    x0 = b.clone()
    y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
    chi = 4 / (1 + math.sqrt(1 + 16 * L**2))

    half_steps = args.ref_steps // 2
    t0 = time.perf_counter()
    xstar, saved, last_rel_step = pdhg_reference(
        prob,
        x0,
        y0,
        L,
        args.ref_steps,
        checkpoints=[half_steps],
        tau=args.pdhg_tau,
    )
    ref_time = time.perf_counter() - t0
    xhalf = saved[half_steps]
    nx = torch.sqrt((xstar**2).sum(dim=(1, 2, 3))).clamp_min(1e-12)
    ref_self_gap = core.rel_error_vec(xhalf, xstar, nx).detach().cpu().numpy()

    xold = core.unroll_final(prob, x0, y0, args.old_ref_iters, L=L, cfg=cfg, fixed=(chi - 0.05 * chi, 0.10))
    old_gap = core.rel_error_vec(xold, xstar, nx).detach().cpu().numpy()

    net = core.PrecNet().to(DEV)
    net.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    net.eval()

    methods = ["plain-FBHF", "line-search-FBHF", "fixed-precond", "learned-warped-bv"]
    summary_rows = []
    curve_rows = []
    per_image_rows = []
    endpoint_by_method = {}
    for method in methods:
        if DEV == "cuda":
            torch.cuda.synchronize()
        mt0 = time.perf_counter()
        errs, endpoint, oracle_marks, counters = run_fbhf_method(
            prob, x0, y0, L, cfg, args.fbhf_steps, xstar, nx, method, net=net, chi=chi
        )
        if DEV == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - mt0
        endpoint_by_method[method] = endpoint
        row = {
            "data": data,
            "blur": blur,
            "noise": noise,
            "method": method,
            "oracle_K": int(oracle_marks[-1]),
            "wall_time_K": float(elapsed),
            **endpoint_summary(endpoint),
            **counters,
        }
        summary_rows.append(row)
        for i, (err, oracle) in enumerate(zip(errs, oracle_marks), start=1):
            curve_rows.append(
                {
                    "data": data,
                    "blur": blur,
                    "noise": noise,
                    "method": method,
                    "iteration": i,
                    "oracle": int(oracle),
                    "mean_rel_primal_error": float(err),
                    "wall_time_s": float(elapsed * i / len(errs)),
                }
            )
        for j, val in enumerate(endpoint):
            per_image_rows.append(
                {
                    "data": data,
                    "blur": blur,
                    "noise": noise,
                    "method": method,
                    "image_index": j,
                    "rel_primal_error_K": float(val),
                    "oracle_K": int(oracle_marks[-1]),
                }
            )

    pdhg_steps = args.oracle_budget // 3
    if DEV == "cuda":
        torch.cuda.synchronize()
    mt0 = time.perf_counter()
    pdhg_errs, pdhg_endpoint = pdhg_trace(prob, x0, y0, L, pdhg_steps, xstar, nx, tau=args.pdhg_tau)
    if DEV == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - mt0
    endpoint_by_method["Condat-Vu/PDHG"] = pdhg_endpoint
    summary_rows.append(
        {
            "data": data,
            "blur": blur,
            "noise": noise,
            "method": "Condat-Vu/PDHG",
            "oracle_K": int(pdhg_steps * 3),
            "wall_time_K": float(elapsed),
            **endpoint_summary(pdhg_endpoint),
            "prox": pdhg_steps,
            "B": pdhg_steps,
            "C": pdhg_steps,
            "network": 0,
            "backtracks": 0,
        }
    )
    for i, err in enumerate(pdhg_errs, start=1):
        curve_rows.append(
            {
                "data": data,
                "blur": blur,
                "noise": noise,
                "method": "Condat-Vu/PDHG",
                "iteration": i,
                "oracle": int(i * 3),
                "mean_rel_primal_error": float(err),
                "wall_time_s": float(elapsed * i / len(pdhg_errs)),
            }
        )
    for j, val in enumerate(pdhg_endpoint):
        per_image_rows.append(
            {
                "data": data,
                "blur": blur,
                "noise": noise,
                "method": "Condat-Vu/PDHG",
                "image_index": j,
                "rel_primal_error_K": float(val),
                "oracle_K": int(pdhg_steps * 3),
            }
        )

    fixed = endpoint_by_method["fixed-precond"]
    learned = endpoint_by_method["learned-warped-bv"]
    cv = endpoint_by_method["Condat-Vu/PDHG"]
    diagnostics = []
    for j in range(args.ntest):
        diagnostics.append(
            {
                "data": data,
                "blur": blur,
                "noise": noise,
                "image_index": j,
                "ref_self_gap_50k_100k": float(ref_self_gap[j]),
                "old_fbhf4000_gap_to_ref": float(old_gap[j]),
                "last_pdhg_rel_step": float(last_rel_step[j]),
                "fixed_minus_learned_absdiff": float(abs(fixed[j] - learned[j])),
                "fixed_better_than_learned": int(fixed[j] < learned[j]),
                "learned_better_than_fixed": int(learned[j] < fixed[j]),
                "cv_better_than_learned": int(cv[j] < learned[j]),
                "ref_steps": args.ref_steps,
                "ref_time_s": float(ref_time),
            }
        )

    learned_wins = int((learned < fixed).sum())
    cv_wins = int((cv < learned).sum())
    meta = {
        "data": data,
        "blur": blur,
        "noise": noise,
        "device": DEV,
        "L": L,
        "chi": chi,
        "ref_steps": args.ref_steps,
        "old_ref_iters": args.old_ref_iters,
        "ref_time_s": ref_time,
        "ref_self_gap_mean": float(ref_self_gap.mean()),
        "ref_self_gap_median": float(np.median(ref_self_gap)),
        "ref_self_gap_max": float(ref_self_gap.max()),
        "old_fbhf4000_gap_mean": float(old_gap.mean()),
        "old_fbhf4000_gap_median": float(np.median(old_gap)),
        "old_fbhf4000_gap_max": float(old_gap.max()),
        "last_pdhg_rel_step_mean": float(last_rel_step.mean()),
        "last_pdhg_rel_step_max": float(last_rel_step.max()),
        "learned_wins_vs_fixed": learned_wins,
        "learned_vs_fixed_sign_p": sign_test_p_value(learned_wins, args.ntest),
        "cv_wins_vs_learned": cv_wins,
        "cv_vs_learned_sign_p": sign_test_p_value(cv_wins, args.ntest),
        "fixed_minus_learned_absdiff_mean": float(np.abs(fixed - learned).mean()),
        "fixed_minus_learned_absdiff_median": float(np.median(np.abs(fixed - learned))),
    }
    return summary_rows, curve_rows, per_image_rows, diagnostics, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(CORE_DIR / "wbv_seed11.pt"))
    parser.add_argument("--image_dir", default=str(CORE_DIR / "standard_images"))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--test_seed", type=int, default=20240704)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--ntest", type=int, default=64)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--fbhf_steps", type=int, default=1000)
    parser.add_argument("--oracle_budget", type=int, default=4000)
    parser.add_argument("--ref_steps", type=int, default=100000)
    parser.add_argument("--old_ref_iters", type=int, default=4000)
    parser.add_argument("--pdhg_tau", type=float, default=1.0)
    parser.add_argument("--conditions", default="real:train:train,synthetic:train:train")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    core.download_images(args.image_dir)

    all_summary = []
    all_curves = []
    all_per_image = []
    all_diag = []
    metas = []
    for condition in args.conditions.split(","):
        data, blur, noise = condition.split(":")
        print(f"[high-ref] {data}/{blur}/{noise}", flush=True)
        summary, curves, per_image, diag, meta = evaluate_condition(args, data, blur, noise)
        all_summary.extend(summary)
        all_curves.extend(curves)
        all_per_image.extend(per_image)
        all_diag.extend(diag)
        metas.append(meta)
        (outdir / "meta_partial.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")

    write_csv(
        outdir / "summary_high_precision.csv",
        all_summary,
        [
            "data",
            "blur",
            "noise",
            "method",
            "oracle_K",
            "wall_time_K",
            "err_K_mean",
            "err_K_std",
            "err_K_median",
            "prox",
            "B",
            "C",
            "network",
            "backtracks",
        ],
    )
    write_csv(
        outdir / "curves_high_precision.csv",
        all_curves,
        ["data", "blur", "noise", "method", "iteration", "oracle", "mean_rel_primal_error", "wall_time_s"],
    )
    write_csv(
        outdir / "per_image_high_precision.csv",
        all_per_image,
        ["data", "blur", "noise", "method", "image_index", "rel_primal_error_K", "oracle_K"],
    )
    write_csv(
        outdir / "reference_diagnostics_high_precision.csv",
        all_diag,
        [
            "data",
            "blur",
            "noise",
            "image_index",
            "ref_self_gap_50k_100k",
            "old_fbhf4000_gap_to_ref",
            "last_pdhg_rel_step",
            "fixed_minus_learned_absdiff",
            "fixed_better_than_learned",
            "learned_better_than_fixed",
            "cv_better_than_learned",
            "ref_steps",
            "ref_time_s",
        ],
    )
    (outdir / "meta_high_precision.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
    plot_ref_gap(outdir, all_diag)
    print(json.dumps(metas, indent=2), flush=True)
    print("[done]", outdir, flush=True)


if __name__ == "__main__":
    main()
