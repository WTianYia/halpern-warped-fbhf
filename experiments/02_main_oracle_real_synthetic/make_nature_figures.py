import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)

COLORS = {
    "plain-FBHF": "#6E6E6E",
    "fixed-precond": "#3775BA",
    "learned-warped(ours)": "#B64342",
    "Halpern-FBHF": "#B64342",
    "bv": "#B64342",
    "free": "#9A4D8E",
}


def panel_label(ax, label, x=-0.12, y=1.04):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def save_all(fig, outbase):
    outbase = Path(outbase)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_curve_csv(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["iteration"] = int(row["iteration"])
            row["median_rel_primal_error"] = float(row["median_rel_primal_error"])
            rows.append(row)
    return rows


def read_metrics(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)["metrics"]


def plot_warped(package_dir, outdir):
    package_dir = Path(package_dir)
    outdir = Path(outdir)
    bv_curves = read_curve_csv(package_dir / "warped_main" / "curves_bv.csv")
    free_curves = read_curve_csv(package_dir / "warped_main" / "curves_free.csv")
    bv_metrics = read_metrics(package_dir / "warped_main" / "metrics_bv.json")
    free_metrics = read_metrics(package_dir / "warped_main" / "metrics_free.json")

    fig = plt.figure(figsize=(7.2, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.36)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    grouped = defaultdict(list)
    for row in bv_curves:
        grouped[row["method"]].append(row)
    order = ["plain-FBHF", "fixed-precond", "learned-warped(ours)"]
    for method in order:
        data = sorted(grouped[method], key=lambda r: r["iteration"])
        x = [r["iteration"] for r in data]
        y = [r["median_rel_primal_error"] for r in data]
        lw = 1.8 if method == "learned-warped(ours)" else 1.35
        ax0.semilogy(x, y, color=COLORS[method], lw=lw, label=method)
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Median relative error")
    ax0.set_xlim(0, 400)
    ax0.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.85)
    ax0.legend(loc="upper right", fontsize=6.4, handlelength=1.5)
    ax0.set_title("Theory-safe learned metric", fontsize=7.5, pad=5)
    panel_label(ax0, "a")

    methods = order
    x = np.arange(len(methods))
    width = 0.34
    bv_vals = [bv_metrics[m]["err_K"] for m in methods]
    free_vals = [free_metrics[m]["err_K"] for m in methods]
    ax1.bar(
        x - width / 2,
        bv_vals,
        width,
        color=[COLORS[m] for m in methods],
        edgecolor="black",
        linewidth=0.45,
        label="BV (provable)",
    )
    ax1.bar(
        x + width / 2,
        free_vals,
        width,
        color=[COLORS[m] for m in methods],
        edgecolor="black",
        linewidth=0.45,
        hatch="///",
        alpha=0.8,
        label="Free metric",
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Error at 400 iters")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Plain", "Fixed\nmetric", "Learned\nmetric"])
    ax1.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.85)
    ax1.legend(fontsize=6.2, loc="upper right")
    ax1.set_title("Constraint cost", fontsize=7.5, pad=5)
    panel_label(ax1, "b")

    save_all(fig, outdir / "fig1_warped_metric")


def read_nonunique(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for key in [
                "init_id",
                "initial_free_coordinate",
                "iteration",
                "distance_to_min_norm",
                "residual",
                "free_coordinate",
                "z1",
                "z2",
                "z3",
                "z4",
            ]:
                row[key] = float(row[key])
            row["init_id"] = int(row["init_id"])
            row["iteration"] = int(row["iteration"])
            rows.append(row)
    return rows


def read_nonunique_finals(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for key in [
                "init_id",
                "initial_free_coordinate",
                "final_free_coordinate",
                "final_distance_to_min_norm",
                "final_residual",
                "z1",
                "z2",
                "z3",
                "z4",
            ]:
                row[key] = float(row[key])
            row["init_id"] = int(row["init_id"])
            rows.append(row)
    return rows


def plot_nonunique(package_dir, outdir):
    package_dir = Path(package_dir)
    outdir = Path(outdir)
    curves = read_nonunique(package_dir / "nonunique_selection" / "nonunique_curves.csv")
    finals = read_nonunique_finals(package_dir / "nonunique_selection" / "nonunique_finals.csv")

    fig = plt.figure(figsize=(7.2, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.36)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    for method in ["plain-FBHF", "Halpern-FBHF"]:
        by_iter = defaultdict(list)
        for row in curves:
            if row["method"] == method:
                by_iter[row["iteration"]].append(row["distance_to_min_norm"])
        xs = sorted(by_iter.keys())
        mean = np.array([np.mean(by_iter[i]) for i in xs])
        lo = np.array([np.min(by_iter[i]) for i in xs])
        hi = np.array([np.max(by_iter[i]) for i in xs])
        color = COLORS.get(method, "#6E6E6E")
        ax0.semilogy(xs, mean, color=color, lw=1.8 if method == "Halpern-FBHF" else 1.35, label=method)
        ax0.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Distance to minimum-norm solution")
    ax0.set_xlim(0, 3000)
    ax0.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.85)
    ax0.legend(loc="upper right", fontsize=6.4)
    ax0.set_title("Selection by anchoring", fontsize=7.5, pad=5)
    panel_label(ax0, "a")

    for method, marker, offset in [("plain-FBHF", "o", -0.035), ("Halpern-FBHF", "s", 0.035)]:
        data = [r for r in finals if r["method"] == method]
        xs = np.array([r["initial_free_coordinate"] for r in data]) + offset
        ys = np.array([r["final_free_coordinate"] for r in data])
        ax1.scatter(
            xs,
            ys,
            s=30,
            marker=marker,
            color=COLORS.get(method, "#6E6E6E"),
            edgecolor="black",
            linewidth=0.45,
            label=method,
            zorder=3,
        )
    lim = 3.35
    ax1.plot([-lim, lim], [-lim, lim], color="#A8A8A8", lw=0.8, ls="--")
    ax1.axhline(0, color="#272727", lw=0.65)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("Initial free coordinate")
    ax1.set_ylabel("Final free coordinate")
    ax1.grid(color="#D8D8D8", linewidth=0.45, alpha=0.75)
    ax1.legend(loc="upper left", fontsize=6.2)
    ax1.set_title("Initial-state dependence", fontsize=7.5, pad=5)
    panel_label(ax1, "b")

    save_all(fig, outdir / "fig2_halpern_selection")


def write_contract(package_dir, outdir):
    text = """# Figure contract

Core conclusion:
Learned warped/preconditioned FBHF gives a theory-safe fixed-budget acceleration, while Halpern anchoring selects the minimum-norm solution when the solution set is nonunique.

Figure archetype:
Quantitative grid.

Backend:
Python / matplotlib only.

Final size:
Double-column manuscript figures, about 183 mm wide.

Panel map:
- Fig. 1a: median relative primal error for the provable BV learned metric.
- Fig. 1b: endpoint error comparison between BV and free metric variants.
- Fig. 2a: distance to the minimum-norm solution in a nonunique three-operator problem.
- Fig. 2b: final free coordinate vs initial free coordinate.

Evidence hierarchy:
- Hero evidence: Fig. 1a and Fig. 2a.
- Validation evidence: Fig. 1b and Fig. 2b.
- Controls: plain-FBHF, fixed-preconditioned FBHF, free-metric comparison.

Statistics:
TV curves report median relative primal error over held-out synthetic TV deblurring instances. Nonunique curves report mean with min-max band across four initial points.

Source data:
All plotted values are read from CSV/JSON files in the experiment package.

Reviewer risk:
Current TV experiment is one trained seed and one held-out batch; final submission should add additional image kernels/noise levels or seeds if the target journal expects broad numerical benchmarking.
"""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "FIGURE_CONTRACT.md").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package_dir", default="final_experiment_package")
    ap.add_argument("--outdir", default="final_experiment_package/figures_nature")
    args = ap.parse_args()
    write_contract(args.package_dir, args.outdir)
    plot_warped(args.package_dir, args.outdir)
    plot_nonunique(args.package_dir, args.outdir)
    print("saved figures to", args.outdir)


if __name__ == "__main__":
    main()
