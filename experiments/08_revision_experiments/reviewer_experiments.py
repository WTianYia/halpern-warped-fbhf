import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = ROOT / "experiments" / "02_main_oracle_real_synthetic"
sys.path.insert(0, str(CORE_DIR))

import trackB_formal_eval as core  # noqa: E402

DEV = core.DEV


def build_cfg(c, L):
    return dict(tlo=0.1, thi=1.8, slo=0.02, shi=0.12, rho=0.9, tau0=0.3, s0=0.1, c=c)


@torch.no_grad()
def pdhg_step(prob, x, y, xbar, tau, sigma, theta):
    y_new = core.projb(y + sigma * core.Dop(xbar), prob.mu)
    grad = core.Ktf(core.Kf(x, prob.otf) - prob.b, prob.otf)
    x_new = x - tau * (grad + core.Dtop(y_new))
    xbar_new = x_new + theta * (x_new - x)
    return x_new, y_new, xbar_new


@torch.no_grad()
def pdhg_trace(prob, x0, y0, steps, xstar, nx, L, tau=1.0, theta=1.0):
    sigma = 0.99 / (tau * L**2)
    x, y, xbar = x0.clone(), y0.clone(), x0.clone()
    errs = []
    for _ in range(steps):
        x, y, xbar = pdhg_step(prob, x, y, xbar, tau, sigma, theta)
        errs.append(core.rel_error(x, xstar, nx))
    endpoint = core.rel_error_vec(x, xstar, nx).detach().cpu().numpy().tolist()
    counters = {"prox": steps, "B": steps, "C": steps, "network": 0, "backtracks": 0}
    return errs, endpoint, counters


@torch.no_grad()
def pdhg_final(prob, x0, y0, steps, L, tau=1.0, theta=1.0):
    sigma = 0.99 / (tau * L**2)
    x, y, xbar = x0.clone(), y0.clone(), x0.clone()
    for _ in range(steps):
        x, y, xbar = pdhg_step(prob, x, y, xbar, tau, sigma, theta)
    return x


def op_normsq_estimate(tau, s, power_iters=8):
    v = torch.randn_like(tau)
    v = v / torch.sqrt((v**2).sum(dim=(1, 2, 3), keepdim=True)).clamp_min(1e-12)
    sqrt_tau = torch.sqrt(tau.clamp_min(1e-12))
    sqrt_s = torch.sqrt(s.clamp_min(1e-12))
    for _ in range(power_iters):
        q = sqrt_s * core.Dop(sqrt_tau * v)
        v = sqrt_tau * core.Dtop(sqrt_s * q)
        v = v / torch.sqrt((v**2).sum(dim=(1, 2, 3), keepdim=True)).clamp_min(1e-12)
    q = sqrt_s * core.Dop(sqrt_tau * v)
    return (q**2).sum(dim=(1, 2, 3))


@torch.no_grad()
def learned_opcap_trace(prob, x0, y0, steps, xstar, nx, net, L, cfg, cap_power_iters=8):
    x, y, x_prev = x0.clone(), y0.clone(), x0.clone()
    tau = torch.full_like(x, cfg["tau0"])
    s = torch.full_like(x, cfg["s0"])
    errs = []
    cap_rows = []
    counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
    for k in range(steps):
        _, (px0, py0) = core.warped_step(prob, x, y, torch.full_like(x, cfg["tau0"]), torch.full_like(x, cfg["s0"]))
        raw = net(core.features(x, y, px0, py0, x_prev))
        dtau, dsig = core.clamp_ts(raw, L, cfg["tlo"], cfg["thi"], cfg["slo"], cfg["shi"], cfg["rho"])
        counters["network"] += 1
        eta = cfg["c"] / ((k + 1) ** 1.1)
        tau = torch.clamp(tau + torch.clamp(dtau - tau, -eta, eta), cfg["tlo"], cfg["thi"])
        s = torch.clamp(s + torch.clamp(dsig - s, -eta, eta), cfg["slo"], cfg["shi"])
        s = torch.minimum(s, cfg["rho"] / (tau * L**2 + 1e-9))

        before = op_normsq_estimate(tau, s, power_iters=cap_power_iters)
        scale = torch.minimum(torch.ones_like(before), (cfg["rho"] / before.clamp_min(1e-12))).view(-1, 1, 1, 1)
        s = s * scale
        after = op_normsq_estimate(tau, s, power_iters=cap_power_iters)
        cap_rows.append(
            {
                "iteration": k + 1,
                "max_op_normsq_before": float(before.max().item()),
                "mean_op_normsq_before": float(before.mean().item()),
                "max_op_normsq_after": float(after.max().item()),
                "mean_op_normsq_after": float(after.mean().item()),
                "min_scale": float(scale.min().item()),
                "mean_scale": float(scale.mean().item()),
            }
        )

        (x_new, y_new), _ = core.warped_step(prob, x, y, tau, s)
        counters["prox"] += 1
        counters["B"] += 2
        counters["C"] += 1
        x_prev = x
        x, y = x_new, y_new
        errs.append(core.rel_error(x, xstar, nx))
    endpoint = core.rel_error_vec(x, xstar, nx).detach().cpu().numpy().tolist()
    return errs, endpoint, counters, cap_rows


def summarise_endpoint(data, blur, noise, mode, c, method, endpoint, counters, wall_time, oracle):
    arr = np.asarray(endpoint, dtype=float)
    return {
        "data": data,
        "blur": blur,
        "noise": noise,
        "mode": mode,
        "c": c,
        "method": method,
        "err_K": float(arr.mean()),
        "err_K_mean": float(arr.mean()),
        "err_K_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "err_K_median": float(np.median(arr)),
        "oracle_K": int(oracle),
        "wall_time_K": float(wall_time),
        **counters,
    }


@torch.no_grad()
def evaluate_condition(args, data, blur, noise, c):
    core.set_seed(args.seed)
    H = W = args.size
    L = core.normD(H, W, DEV)
    cfg = build_cfg(c, L)
    b, otf, mu, _ = core.make_batch(args.ntest, H, W, DEV, args.test_seed, data, blur, noise, args.image_dir)
    prob = core.TV(b, otf, mu)
    x0 = b.clone()
    y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
    chi = 4 / (1 + math.sqrt(1 + 16 * L**2))

    xstar_fbhf = core.unroll_final(prob, x0, y0, args.ref_iters, L=L, cfg=cfg, fixed=(chi - 0.05 * chi, 0.10))
    xstar_pdhg = pdhg_final(prob, x0, y0, args.pdhg_ref_iters, L=L, tau=args.pdhg_tau)
    nx = torch.sqrt((xstar_fbhf**2).sum(dim=(1, 2, 3))).clamp_min(1e-12)
    nx_pdhg = torch.sqrt((xstar_pdhg**2).sum(dim=(1, 2, 3))).clamp_min(1e-12)
    ref_gap = core.rel_error_vec(xstar_fbhf, xstar_pdhg, nx_pdhg).detach().cpu().numpy()
    ref_obj_gap = (prob.obj(xstar_fbhf) - prob.obj(xstar_pdhg)).detach().cpu().numpy()

    net = core.PrecNet().to(DEV)
    net.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    net.eval()

    method_specs = []
    method_specs.append(("plain-FBHF", "fbhf", dict(fixed=(chi - 0.05 * chi, 0.10), net=None, mode="bv")))
    method_specs.append(("line-search-FBHF", "linesearch", {}))
    method_specs.append(("fixed-precond", "fbhf", dict(fixed=(1.5, 0.07), net=None, mode="bv")))
    method_specs.append((f"learned-warped-bv-c{c:g}", "fbhf", dict(fixed=None, net=net, mode="bv")))
    method_specs.append((f"learned-warped-opcap-c{c:g}", "opcap", {}))
    method_specs.append(("Condat-Vu/PDHG", "pdhg", {}))

    summary, curves, per_image, cap_rows = [], [], [], []
    budget = args.oracle_budget
    for method, kind, spec in method_specs:
        if DEV == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if kind == "linesearch":
            steps = args.fbhf_steps
            errs, endpoint, oracle_marks, counters = core.unroll_linesearch_trace(prob, x0, y0, steps, chi=chi, xstar=xstar_fbhf, nx=nx)
            oracle_series = oracle_marks
        elif kind == "pdhg":
            steps = budget // 3
            errs, endpoint, counters = pdhg_trace(prob, x0, y0, steps, xstar_fbhf, nx, L, tau=args.pdhg_tau)
            oracle_series = [(i + 1) * 3 for i in range(len(errs))]
        elif kind == "opcap":
            steps = args.fbhf_steps
            errs, endpoint, counters, cap_rows_method = learned_opcap_trace(
                prob, x0, y0, steps, xstar_fbhf, nx, net, L, cfg, cap_power_iters=args.cap_power_iters
            )
            oracle_series = [(i + 1) * 4 for i in range(len(errs))]
            for row in cap_rows_method:
                row.update({"data": data, "blur": blur, "noise": noise, "c": c, "method": method})
            cap_rows.extend(cap_rows_method)
        else:
            steps = args.fbhf_steps
            counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
            errs, endpoint = core.unroll_trace(
                prob,
                x0,
                y0,
                steps,
                xstar_fbhf,
                nx,
                net=spec["net"],
                mode=spec["mode"],
                L=L,
                cfg=cfg,
                fixed=spec["fixed"],
                counters=counters,
            )
            oracle_series = [(i + 1) * 4 for i in range(len(errs))]
        if DEV == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        endpoint_arr = np.asarray(endpoint, dtype=float)
        oracle_K = oracle_series[-1]
        summary.append(summarise_endpoint(data, blur, noise, "bv", c, method, endpoint, counters, elapsed, oracle_K))
        for i, err in enumerate(errs):
            curves.append(
                {
                    "data": data,
                    "blur": blur,
                    "noise": noise,
                    "c": c,
                    "method": method,
                    "iteration": i + 1,
                    "oracle": oracle_series[i],
                    "mean_rel_primal_error": float(err),
                }
            )
        for j, val in enumerate(endpoint_arr):
            per_image.append(
                {
                    "data": data,
                    "blur": blur,
                    "noise": noise,
                    "c": c,
                    "method": method,
                    "image_index": j,
                    "oracle_K": oracle_K,
                    "rel_primal_error_K": float(val),
                }
            )

    ref_rows = []
    for j, val in enumerate(ref_gap):
        ref_rows.append(
            {
                "data": data,
                "blur": blur,
                "noise": noise,
                "image_index": j,
                "fbhf_vs_pdhg_ref_rel_gap": float(val),
                "fbhf_minus_pdhg_obj": float(ref_obj_gap[j]),
            }
        )
    return summary, curves, per_image, cap_rows, ref_rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_audit(path):
    audit = {
        "current_formal_train_default": {
            "data": "real",
            "image_source_pool": sorted(core.IMAGE_URLS.keys()),
            "source_level_split": False,
            "note": "The existing checkpoint is trained on random augmented patches drawn from the same standard-image pool used for real-image evaluation. Exact test patches are held out by random seed, but the protocol is patch-level rather than source-image-level.",
        },
        "recommended_revision_protocol": {
            "source_level_train": ["camera", "coffee", "astronaut", "coins", "page", "moon", "chelsea", "sails"],
            "source_level_test": ["barbara", "boat", "peppers", "cameraman_classic", "goldhill", "baboon"],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--image_dir", default=str(CORE_DIR / "standard_images"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--test_seed", type=int, default=20240704)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--ntest", type=int, default=64)
    ap.add_argument("--fbhf_steps", type=int, default=1000)
    ap.add_argument("--oracle_budget", type=int, default=4000)
    ap.add_argument("--ref_iters", type=int, default=4000)
    ap.add_argument("--pdhg_ref_iters", type=int, default=8000)
    ap.add_argument("--pdhg_tau", type=float, default=1.0)
    ap.add_argument("--cap_power_iters", type=int, default=6)
    ap.add_argument("--c_values", default="0.5")
    ap.add_argument("--conditions", default="real:train:train,synthetic:train:train")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    core.download_images(args.image_dir)
    all_summary, all_curves, all_per_image, all_caps, all_refs = [], [], [], [], []
    for cond in args.conditions.split(","):
        data, blur, noise = cond.split(":")
        for c in [float(x) for x in args.c_values.split(",")]:
            print("[revision-eval]", data, blur, noise, "c", c, flush=True)
            summary, curves, per_image, caps, refs = evaluate_condition(args, data, blur, noise, c)
            all_summary.extend(summary)
            all_curves.extend(curves)
            all_per_image.extend(per_image)
            all_caps.extend(caps)
            all_refs.extend(refs)
            (outdir / "summary_partial.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")

    (outdir / "summary_revision.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    write_csv(
        outdir / "curves_revision.csv",
        all_curves,
        ["data", "blur", "noise", "c", "method", "iteration", "oracle", "mean_rel_primal_error"],
    )
    write_csv(
        outdir / "per_image_revision.csv",
        all_per_image,
        ["data", "blur", "noise", "c", "method", "image_index", "oracle_K", "rel_primal_error_K"],
    )
    write_csv(
        outdir / "operator_cap_revision.csv",
        all_caps,
        [
            "data",
            "blur",
            "noise",
            "c",
            "method",
            "iteration",
            "max_op_normsq_before",
            "mean_op_normsq_before",
            "max_op_normsq_after",
            "mean_op_normsq_after",
            "min_scale",
            "mean_scale",
        ],
    )
    write_csv(
        outdir / "reference_check_revision.csv",
        all_refs,
        ["data", "blur", "noise", "image_index", "fbhf_vs_pdhg_ref_rel_gap", "fbhf_minus_pdhg_obj"],
    )
    write_audit(outdir / "source_split_audit.json")
    print("[done]", outdir, flush=True)


if __name__ == "__main__":
    main()

