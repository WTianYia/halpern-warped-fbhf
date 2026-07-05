import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = ROOT / "experiments" / "02_main_oracle_real_synthetic"
REV_DIR = ROOT / "experiments" / "08_revision_experiments"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(REV_DIR))

import trackB_formal_eval as core  # noqa: E402
from reviewer_experiments import pdhg_trace, pdhg_final  # noqa: E402

DEV = core.DEV

TRAIN_SOURCES = ["camera", "coffee", "astronaut", "coins", "page", "moon", "chelsea", "sails"]
TEST_SOURCES = ["barbara", "boat", "peppers", "cameraman_classic", "goldhill", "baboon"]


def build_cfg(c, L):
    return dict(tlo=0.1, thi=1.8, slo=0.02, shi=0.12, rho=0.9, tau0=0.3, s0=0.1, c=c)


def named_image_batch(B, H, W, seed, names, image_dir):
    core.download_images(image_dir)
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(B):
        name = names[int(rng.integers(0, len(names)))]
        path = Path(image_dir) / f"{name}.png"
        x = core.pil_to_gray_tensor(path, H, W)
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[-1])
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[-2])
        gain = float(rng.uniform(0.85, 1.15))
        bias = float(rng.uniform(-0.05, 0.05))
        imgs.append((x * gain + bias).clamp(0, 1))
    return torch.cat(imgs, dim=0)


def make_batch_sources(B, H, W, device, seed, sources, blur="train", noise="train", image_dir="standard_images"):
    imgs = named_image_batch(B, H, W, seed, sources, image_dir).to(device)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 17)
    if blur == "train":
        sigma = (torch.rand(B, generator=g) * 1.5 + 1.0).to(device)
        otf = core.gauss_otf(H, W, sigma, device)
    else:
        sigma = torch.full((B,), float(blur), device=device)
        otf = core.gauss_otf(H, W, sigma, device)
    if noise == "train":
        sig = (torch.rand(B, 1, 1, 1, generator=g) * 0.015 + 0.005).to(device)
    elif noise == "high":
        sig = torch.full((B, 1, 1, 1), 0.03, device=device)
    elif noise == "low":
        sig = torch.full((B, 1, 1, 1), 0.005, device=device)
    else:
        sig = torch.full((B, 1, 1, 1), float(noise), device=device)
    eps = torch.randn(B, 1, H, W, generator=g).to(device)
    b = (core.Kf(imgs, otf) + sig * eps).clamp(0, 1)
    return b, otf, 0.02, imgs


def objective_after(prob, xs):
    weights = torch.arange(1, len(xs) + 1, device=xs[0].device, dtype=xs[0].dtype)
    vals = torch.stack([prob.obj(x).mean() for x in xs])
    return (weights * vals).sum() / weights.sum()


@torch.no_grad()
def val_objective_source(net, args, L, cfg, seed):
    b, otf, mu, _ = make_batch_sources(args.batch, args.size, args.size, DEV, seed, TEST_SOURCES, args.blur, args.noise, args.image_dir)
    prob = core.TV(b, otf, mu)
    x0 = b.clone()
    y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
    learned = prob.obj(core.unroll(prob, x0, y0, args.train_K, net=net, mode="bv", L=L, cfg=cfg)[-1]).mean().item()
    fixed = prob.obj(core.unroll(prob, x0, y0, args.train_K, L=L, cfg=cfg, fixed=(1.5, 0.07))[-1]).mean().item()
    return learned, fixed


def train_source_split(args, L, cfg):
    core.set_seed(args.seed)
    ckpt = Path(args.source_ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    net = core.PrecNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best = float("inf")
    for it in range(1, args.train_iters + 1):
        b, otf, mu, _ = make_batch_sources(
            args.batch,
            args.size,
            args.size,
            DEV,
            args.seed * 100000 + it,
            TRAIN_SOURCES,
            args.blur,
            args.noise,
            args.image_dir,
        )
        prob = core.TV(b, otf, mu)
        x0 = b.clone()
        y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
        xs = core.unroll(prob, x0, y0, args.train_K, net=net, mode="bv", L=L, cfg=cfg)
        loss = objective_after(prob, xs)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it == 1 or it % args.eval_every == 0:
            net.eval()
            learned, fixed = val_objective_source(net, args, L, cfg, args.val_seed)
            net.train()
            tag = ""
            if math.isfinite(learned) and learned < best:
                best = learned
                torch.save(net.state_dict(), ckpt)
                tag = " *saved"
            print(f"it{it:5d} loss {loss.item():.6e} | source-val learned {learned:.6e} vs fixed {fixed:.6e}{tag}", flush=True)
    print(f"[source-train] best={best:.6e} -> {ckpt}", flush=True)


@torch.no_grad()
def unroll_halpern_trace(prob, x0, y0, steps, xstar, nx, L, cfg, anchor_x=None, anchor_y=None):
    x, y = x0.clone(), y0.clone()
    if anchor_x is None:
        anchor_x = torch.zeros_like(x0)
    if anchor_y is None:
        anchor_y = torch.zeros_like(y0)
    chi = 4 / (1 + math.sqrt(1 + 16 * L**2))
    tau = torch.full_like(x, chi - 0.05 * chi)
    s = torch.full_like(x, 0.10)
    errs = []
    for k in range(steps):
        (tx, ty), _ = core.warped_step(prob, x, y, tau, s)
        lam = 1.0 / (k + 2.0)
        x = lam * anchor_x + (1.0 - lam) * tx
        y = lam * anchor_y + (1.0 - lam) * ty
        errs.append(core.rel_error(x, xstar, nx))
    endpoint = core.rel_error_vec(x, xstar, nx).detach().cpu().numpy().tolist()
    counters = {"prox": steps, "B": 2 * steps, "C": steps, "network": 0, "backtracks": 0}
    return errs, endpoint, counters


@torch.no_grad()
def inertial_trace(prob, x0, y0, steps, xstar, nx, tau_value, s_value, alpha):
    x, y = x0.clone(), y0.clone()
    xp, yp = x.clone(), y.clone()
    tau = torch.full_like(x, tau_value)
    s = torch.full_like(x, s_value)
    errs = []
    diverged = False
    for _ in range(steps):
        wx = x + alpha * (x - xp)
        wy = y + alpha * (y - yp)
        (tx, ty), _ = core.warped_step(prob, wx, wy, tau, s)
        xp, yp = x, y
        x, y = tx, ty
        if not torch.isfinite(x).all() or not torch.isfinite(y).all() or prob.obj(x).mean().item() > 1e8:
            diverged = True
            break
        errs.append(core.rel_error(x, xstar, nx))
    if diverged or not errs:
        endpoint = [float("inf")] * x0.shape[0]
    else:
        endpoint = core.rel_error_vec(x, xstar, nx).detach().cpu().numpy().tolist()
    counters = {"prox": len(errs), "B": 2 * len(errs), "C": len(errs), "network": 0, "backtracks": 0}
    return errs, endpoint, counters, diverged


def summarise(data, method, endpoint, counters, oracle, wall):
    arr = np.asarray(endpoint, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        mean = std = median = float("inf")
    else:
        mean = float(finite.mean())
        std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        median = float(np.median(finite))
    return {
        "data": data,
        "method": method,
        "err_K_mean": mean,
        "err_K_std": std,
        "err_K_median": median,
        "oracle_K": int(oracle),
        "wall_time_K": float(wall),
        **counters,
    }


@torch.no_grad()
def evaluate_source_split(args, L, cfg, net):
    b, otf, mu, _ = make_batch_sources(args.ntest, args.size, args.size, DEV, args.test_seed, TEST_SOURCES, args.blur, args.noise, args.image_dir)
    prob = core.TV(b, otf, mu)
    x0 = b.clone()
    y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
    chi = 4 / (1 + math.sqrt(1 + 16 * L**2))
    xstar = core.unroll_final(prob, x0, y0, args.ref_iters, L=L, cfg=cfg, fixed=(chi - 0.05 * chi, 0.10))
    xstar_pdhg = pdhg_final(prob, x0, y0, args.pdhg_ref_iters, L=L, tau=args.pdhg_tau)
    nx = torch.sqrt((xstar**2).sum(dim=(1, 2, 3))).clamp_min(1e-12)

    summary = []
    curves = []
    per_image = []
    methods = [
        ("plain-FBHF", "plain"),
        ("Halpern-anchored-FBHF", "halpern"),
        ("line-search-FBHF", "linesearch"),
        ("fixed-precond", "fixed"),
        ("learned-source-split", "learned"),
        ("Condat-Vu/PDHG", "pdhg"),
    ]
    for method, kind in methods:
        if DEV == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if kind == "plain":
            counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
            errs, endpoint = core.unroll_trace(prob, x0, y0, args.fbhf_steps, xstar, nx, L=L, cfg=cfg, fixed=(chi - 0.05 * chi, 0.10), counters=counters)
            oracle_series = [(i + 1) * 4 for i in range(len(errs))]
        elif kind == "halpern":
            errs, endpoint, counters = unroll_halpern_trace(prob, x0, y0, args.fbhf_steps, xstar, nx, L, cfg)
            oracle_series = [(i + 1) * 4 for i in range(len(errs))]
        elif kind == "linesearch":
            errs, endpoint, oracle_series, counters = core.unroll_linesearch_trace(prob, x0, y0, args.fbhf_steps, chi=chi, xstar=xstar, nx=nx)
        elif kind == "fixed":
            counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
            errs, endpoint = core.unroll_trace(prob, x0, y0, args.fbhf_steps, xstar, nx, L=L, cfg=cfg, fixed=(1.5, 0.07), counters=counters)
            oracle_series = [(i + 1) * 4 for i in range(len(errs))]
        elif kind == "learned":
            counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
            errs, endpoint = core.unroll_trace(prob, x0, y0, args.fbhf_steps, xstar, nx, net=net, mode="bv", L=L, cfg=cfg, fixed=None, counters=counters)
            oracle_series = [(i + 1) * 4 for i in range(len(errs))]
        else:
            steps = args.oracle_budget // 3
            errs, endpoint, counters = pdhg_trace(prob, x0, y0, steps, xstar, nx, L, tau=args.pdhg_tau)
            oracle_series = [(i + 1) * 3 for i in range(len(errs))]
        if DEV == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - start
        summary.append(summarise("source-test", method, endpoint, counters, oracle_series[-1], wall))
        for i, err in enumerate(errs):
            curves.append({"data": "source-test", "method": method, "iteration": i + 1, "oracle": oracle_series[i], "mean_rel_primal_error": float(err)})
        for i, val in enumerate(endpoint):
            per_image.append({"data": "source-test", "method": method, "image_index": i, "oracle_K": oracle_series[-1], "rel_primal_error_K": float(val)})

    ref_gap = core.rel_error_vec(xstar, xstar_pdhg, torch.sqrt((xstar_pdhg**2).sum(dim=(1, 2, 3))).clamp_min(1e-12)).detach().cpu().numpy()
    return summary, curves, per_image, [{"mean_ref_gap": float(ref_gap.mean()), "max_ref_gap": float(ref_gap.max())}], prob, x0, y0, xstar, nx


@torch.no_grad()
def tune_inertial(args, L, cfg, prob, x0, y0, xstar, nx):
    chi = 4 / (1 + math.sqrt(1 + 16 * L**2))
    grid = []
    best = None
    for alpha in args.inertial_alphas:
        for tau_factor in args.inertial_tau_factors:
            for s_value in args.inertial_s_values:
                tau_value = tau_factor * chi
                errs, endpoint, counters, diverged = inertial_trace(prob, x0, y0, args.fbhf_steps, xstar, nx, tau_value, s_value, alpha)
                arr = np.asarray(endpoint, dtype=float)
                mean = float(arr[np.isfinite(arr)].mean()) if np.isfinite(arr).any() else float("inf")
                row = {
                    "alpha": alpha,
                    "tau_factor": tau_factor,
                    "tau": tau_value,
                    "s": s_value,
                    "err_K_mean": mean,
                    "diverged": bool(diverged),
                    **counters,
                }
                grid.append(row)
                if math.isfinite(mean) and (best is None or mean < best["err_K_mean"]):
                    best = row
    return grid, best


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(outdir):
    outdir = Path(outdir)
    with (outdir / "source_split_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)["summary"]
    curves = []
    with (outdir / "source_split_curves.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["oracle"] = float(row["oracle"])
            row["mean_rel_primal_error"] = float(row["mean_rel_primal_error"])
            curves.append(row)

    colors = {
        "plain-FBHF": "#707070",
        "Halpern-anchored-FBHF": "#9B59B6",
        "line-search-FBHF": "#9B9B9B",
        "best-tuned-inertial-FBHF": "#D9903D",
        "fixed-precond": "#3B73B9",
        "learned-source-split": "#B0443F",
        "Condat-Vu/PDHG": "#2E7D59",
    }
    labels = {
        "plain-FBHF": "Plain FBHF",
        "Halpern-anchored-FBHF": "Halpern-anchored FBHF",
        "line-search-FBHF": "Line-search FBHF",
        "best-tuned-inertial-FBHF": "Best inertial FBHF",
        "fixed-precond": "Fixed warped FBHF",
        "learned-source-split": "Learned warped, source split",
        "Condat-Vu/PDHG": "Condat--Vu / PDHG",
    }
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    })
    fig = plt.figure(figsize=(7.0, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    for method in ["plain-FBHF", "Halpern-anchored-FBHF", "best-tuned-inertial-FBHF", "fixed-precond", "learned-source-split", "Condat-Vu/PDHG"]:
        rows = [r for r in curves if r["method"] == method]
        rows = sorted(rows, key=lambda r: r["oracle"])
        ax0.semilogy([r["oracle"] for r in rows], [r["mean_rel_primal_error"] for r in rows], color=colors[method], lw=1.7 if method in {"learned-source-split", "Condat-Vu/PDHG"} else 1.2, label=labels[method])
    ax0.set_xlabel("Oracle calls")
    ax0.set_ylabel("Mean relative primal error")
    ax0.grid(axis="y", color="#D8D8D8", linewidth=0.45)
    ax0.legend(fontsize=5.6, loc="upper right")
    ax0.text(-0.13, 1.04, "a", transform=ax0.transAxes, fontweight="bold", fontsize=8)

    methods = ["plain-FBHF", "Halpern-anchored-FBHF", "best-tuned-inertial-FBHF", "fixed-precond", "learned-source-split", "Condat-Vu/PDHG"]
    vals = [next(r for r in summary if r["method"] == m)["err_K_mean"] for m in methods]
    stds = [next(r for r in summary if r["method"] == m)["err_K_std"] for m in methods]
    x = np.arange(len(methods))
    ax1.bar(x, vals, yerr=stds, color=[colors[m] for m in methods], edgecolor="black", linewidth=0.45, error_kw={"elinewidth": 0.65, "capsize": 2})
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Plain", "Halpern", "Inertial", "Fixed", "Learned", "PDHG"], rotation=25, ha="right")
    ax1.set_ylabel("Error at final budget")
    ax1.grid(axis="y", color="#D8D8D8", linewidth=0.45)
    ax1.text(-0.16, 1.04, "b", transform=ax1.transAxes, fontweight="bold", fontsize=8)
    for suffix in [".pdf", ".svg", ".png", ".tiff"]:
        dpi = 600 if suffix == ".tiff" else 300
        fig.savefig(outdir / f"fig_m2_m3_supplement{suffix}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--image_dir", default="standard_images")
    ap.add_argument("--source_ckpt", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--val_seed", type=int, default=9023)
    ap.add_argument("--test_seed", type=int, default=20240723)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ntest", type=int, default=64)
    ap.add_argument("--train_K", type=int, default=40)
    ap.add_argument("--train_iters", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--blur", default="train")
    ap.add_argument("--noise", default="train")
    ap.add_argument("--fbhf_steps", type=int, default=1000)
    ap.add_argument("--oracle_budget", type=int, default=4000)
    ap.add_argument("--ref_iters", type=int, default=4000)
    ap.add_argument("--pdhg_ref_iters", type=int, default=8000)
    ap.add_argument("--pdhg_tau", type=float, default=1.0)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--inertial_alphas", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6])
    ap.add_argument("--inertial_tau_factors", type=float, nargs="+", default=[0.4, 0.6, 0.8, 0.95])
    ap.add_argument("--inertial_s_values", type=float, nargs="+", default=[0.05, 0.10])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    core.set_seed(args.seed)
    L = core.normD(args.size, args.size, DEV)
    cfg = build_cfg(args.c, L)
    if not args.skip_train or not Path(args.source_ckpt).exists():
        train_source_split(args, L, cfg)

    net = core.PrecNet().to(DEV)
    net.load_state_dict(torch.load(args.source_ckpt, map_location=DEV))
    net.eval()

    summary, curves, per_image, ref_rows, prob, x0, y0, xstar, nx = evaluate_source_split(args, L, cfg, net)
    inertial_grid, inertial_best = tune_inertial(args, L, cfg, prob, x0, y0, xstar, nx)
    if inertial_best is not None:
        errs, endpoint, counters, diverged = inertial_trace(prob, x0, y0, args.fbhf_steps, xstar, nx, inertial_best["tau"], inertial_best["s"], inertial_best["alpha"])
        summary.append(summarise("source-test", "best-tuned-inertial-FBHF", endpoint, counters, 4 * len(errs), 0.0))
        for i, err in enumerate(errs):
            curves.append({"data": "source-test", "method": "best-tuned-inertial-FBHF", "iteration": i + 1, "oracle": (i + 1) * 4, "mean_rel_primal_error": float(err)})

    protocol = {
        "train_sources": TRAIN_SOURCES,
        "test_sources": TEST_SOURCES,
        "source_level_split": True,
        "augmentations": ["horizontal flip", "vertical flip", "gain in [0.85,1.15]", "bias in [-0.05,0.05]"],
        "synthetic_piecewise_rule": "Each image draws 4-8 random rectangles, disks, or sinusoidal stripe components with intensities in [0.15,0.95], then clips to [0,1].",
        "tv_weight": 0.02,
        "blur": args.blur,
        "noise": args.noise,
        "size": args.size,
        "ntest": args.ntest,
    }
    with (outdir / "source_split_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "reference_check": ref_rows, "protocol": protocol, "inertial_best": inertial_best}, f, indent=2)
    write_csv(outdir / "source_split_curves.csv", curves)
    write_csv(outdir / "source_split_per_image.csv", per_image)
    write_csv(outdir / "inertial_grid.csv", inertial_grid)
    with (outdir / "source_split_protocol.json").open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)
    make_figures(outdir)
    print(f"[done] {outdir}", flush=True)


if __name__ == "__main__":
    main()
