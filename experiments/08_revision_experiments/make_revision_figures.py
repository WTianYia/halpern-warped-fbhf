import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
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
    "plain-FBHF": "#707070",
    "line-search-FBHF": "#9B9B9B",
    "fixed-precond": "#3B73B9",
    "learned-warped-bv-c0.5": "#B0443F",
    "learned-warped-opcap-c0.5": "#C96B57",
    "Condat-Vu/PDHG": "#2E7D59",
}


LABELS = {
    "plain-FBHF": "Plain FBHF",
    "line-search-FBHF": "Line-search FBHF",
    "fixed-precond": "Fixed warped FBHF",
    "learned-warped-bv-c0.5": "Learned warped FBHF",
    "learned-warped-opcap-c0.5": "Learned warped, op-cap",
    "Condat-Vu/PDHG": "Condat--Vu / PDHG",
}


def panel_label(ax, label):
    ax.text(
        -0.14,
        1.04,
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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_revision_summary(run_dir):
    run_dir = Path(run_dir)
    curves = read_csv(run_dir / "curves_revision.csv")
    per_image = read_csv(run_dir / "per_image_revision.csv")
    opcap = read_csv(run_dir / "operator_cap_revision.csv")

    grouped = defaultdict(list)
    for row in curves:
        row["oracle"] = float(row["oracle"])
        row["mean_rel_primal_error"] = float(row["mean_rel_primal_error"])
        grouped[(row["data"], row["method"])].append(row)

    methods = [
        "plain-FBHF",
        "line-search-FBHF",
        "fixed-precond",
        "learned-warped-bv-c0.5",
        "Condat-Vu/PDHG",
    ]

    fig = plt.figure(figsize=(7.1, 4.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.34)
    ax_real = fig.add_subplot(gs[0, 0])
    ax_syn = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_cap = fig.add_subplot(gs[1, 1])

    for ax, data, title in [(ax_real, "real", "Real-image patches"), (ax_syn, "synthetic", "Synthetic piecewise images")]:
        for method in methods:
            rows = sorted(grouped[(data, method)], key=lambda r: r["oracle"])
            if not rows:
                continue
            lw = 1.75 if method in {"learned-warped-bv-c0.5", "Condat-Vu/PDHG"} else 1.2
            ax.semilogy(
                [r["oracle"] for r in rows],
                [r["mean_rel_primal_error"] for r in rows],
                color=COLORS[method],
                lw=lw,
                label=LABELS[method],
            )
        ax.set_xlabel("Oracle calls")
        ax.set_ylabel("Mean relative primal error")
        ax.set_xlim(0, 4526)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.85)
        ax.set_title(title, fontsize=7.6, pad=5)
    ax_real.legend(loc="upper right", fontsize=5.8, handlelength=1.5)
    panel_label(ax_real, "a")
    panel_label(ax_syn, "b")

    by_case_method = defaultdict(list)
    for row in per_image:
        key = (row["data"], row["method"])
        by_case_method[key].append(float(row["rel_primal_error_K"]))

    bar_methods = ["fixed-precond", "learned-warped-bv-c0.5", "Condat-Vu/PDHG"]
    x = np.arange(2)
    width = 0.23
    offsets = [-width, 0, width]
    for off, method in zip(offsets, bar_methods):
        means = [np.mean(by_case_method[(data, method)]) for data in ["real", "synthetic"]]
        stds = [np.std(by_case_method[(data, method)], ddof=1) for data in ["real", "synthetic"]]
        ax_bar.bar(
            x + off,
            means,
            width,
            yerr=stds,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.45,
            error_kw={"elinewidth": 0.65, "capsize": 2.0, "capthick": 0.65},
            label=LABELS[method],
        )
    ax_bar.set_yscale("log")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["Real", "Synthetic"])
    ax_bar.set_ylabel("Error at final budget")
    ax_bar.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.85)
    panel_label(ax_bar, "c")

    cap_group = defaultdict(list)
    for row in opcap:
        row["iteration"] = int(row["iteration"])
        row["max_op_normsq_before"] = float(row["max_op_normsq_before"])
        row["max_op_normsq_after"] = float(row["max_op_normsq_after"])
        cap_group[row["data"]].append(row)
    for data, ls, title in [("real", "-", "Real"), ("synthetic", "--", "Synthetic")]:
        rows = sorted(cap_group[data], key=lambda r: r["iteration"])
        ax_cap.plot(
            [r["iteration"] for r in rows],
            [r["max_op_normsq_before"] for r in rows],
            color="#B0443F",
            lw=1.2,
            ls=ls,
            label=f"{title}, before projection",
        )
        ax_cap.plot(
            [r["iteration"] for r in rows],
            [r["max_op_normsq_after"] for r in rows],
            color="#2E7D59",
            lw=1.2,
            ls=ls,
            label=f"{title}, after projection",
        )
    ax_cap.axhline(0.9, color="black", lw=0.8, ls=":", label="certified cap")
    ax_cap.set_xlabel("Iteration")
    ax_cap.set_ylabel(r"Estimated $\|M^{-1/2}BM^{-1/2}\|^2$")
    ax_cap.set_ylim(0.55, 0.94)
    ax_cap.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.85)
    ax_cap.legend(loc="lower right", fontsize=5.2, handlelength=1.5)
    panel_label(ax_cap, "d")

    save_all(fig, run_dir / "fig_revision_reviewer_summary")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    plot_revision_summary(args.run_dir)
