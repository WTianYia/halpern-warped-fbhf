from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures_high_precision_robust"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)

COLORS = {
    "plain-FBHF": "#7A7A7A",
    "line-search-FBHF": "#5C8EBB",
    "fixed-precond": "#B57B4A",
    "learned-warped-bv": "#3B8F6A",
}


def read_summary(run: str) -> list[dict[str, str]]:
    path = ROOT / run / "summary_high_precision.csv"
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row(rows: list[dict[str, str]], data: str, blur: str, noise: str, method: str) -> dict[str, str]:
    for r in rows:
        if (
            r["data"] == data
            and r["blur"] == blur
            and r["noise"] == noise
            and r["method"] == method
        ):
            return r
    raise KeyError((data, blur, noise, method))


def save_all(fig: mpl.figure.Figure, name: str) -> None:
    for ext, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", **kwargs)


def write_source(name: str, records: list[dict[str, object]]) -> None:
    path = FIG_DIR / f"source_data_{name}.csv"
    fields = sorted({k for r in records for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def plot_cscan() -> None:
    runs = [(0.25, "high_ref_c025_20260706"), (0.50, "high_ref_run_20260706"), (0.90, "high_ref_c09_20260706")]
    records: list[dict[str, object]] = []
    for c, run_name in runs:
        rows = read_summary(run_name)
        for method in ["fixed-precond", "learned-warped-bv"]:
            r = row(rows, "real", "train", "train", method)
            records.append(
                {
                    "c": c,
                    "method": method,
                    "mean": float(r["err_K_mean"]) * 1e4,
                    "std": float(r["err_K_std"]) * 1e4,
                    "median": float(r["err_K_median"]) * 1e4,
                }
            )

    fig, ax = plt.subplots(figsize=(3.45, 2.20))
    for method, label in [("fixed-precond", "fixed warped"), ("learned-warped-bv", "learned warped")]:
        rs = [r for r in records if r["method"] == method]
        x = np.array([r["c"] for r in rs], dtype=float)
        y = np.array([r["mean"] for r in rs], dtype=float)
        e = np.array([r["std"] for r in rs], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=e,
            color=COLORS[method],
            marker="o",
            markersize=4,
            linewidth=1.3,
            capsize=2,
            label=label,
        )
    ax.set_xlabel("bounded-variation strength c")
    ax.set_ylabel("endpoint error (x10$^{-4}$)")
    ax.set_xticks([0.25, 0.50, 0.90])
    ax.set_xlim(0.18, 0.97)
    ax.set_ylim(bottom=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.56, 1.17),
        ncol=2,
        handlelength=1.5,
        columnspacing=1.0,
    )
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout(pad=0.2)
    save_all(fig, "fig2_cscan")
    plt.close(fig)
    write_source("fig2_cscan", records)


def plot_ood() -> None:
    rows = read_summary("high_ref_ood_20260706")
    settings = [
        ("low\nnoise", "real", "train", "low"),
        ("high\nnoise", "real", "train", "high"),
        ("wide blur\nhigh noise", "real", "wide", "high"),
        ("small blur\nhigh noise", "real", "0.75", "high"),
        ("motion\nGaussian", "real", "motion", "train"),
    ]
    methods = [
        ("plain-FBHF", "plain"),
        ("line-search-FBHF", "line-search"),
        ("learned-warped-bv", "learned"),
    ]
    records: list[dict[str, object]] = []
    for label, data, blur, noise in settings:
        fixed = float(row(rows, data, blur, noise, "fixed-precond")["err_K_mean"])
        for method, short in methods:
            r = row(rows, data, blur, noise, method)
            mean = float(r["err_K_mean"])
            records.append(
                {
                    "setting": label.replace("\n", " "),
                    "method": method,
                    "mean": mean,
                    "std": float(r["err_K_std"]),
                    "ratio_to_fixed": mean / fixed,
                }
            )

    fig, ax = plt.subplots(figsize=(3.55, 2.25))
    x = np.arange(len(settings))
    width = 0.23
    for j, (method, label) in enumerate(methods):
        vals = [
            next(r for r in records if r["setting"] == s[0].replace("\n", " ") and r["method"] == method)[
                "ratio_to_fixed"
            ]
            for s in settings
        ]
        ax.bar(
            x + (j - 1) * width,
            vals,
            width=width,
            color=COLORS[method],
            label=label,
            edgecolor="none",
        )
    ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.002, 12)
    ax.set_ylabel("mean error / fixed warped")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in settings])
    ax.legend(loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.9)
    ax.text(-0.10, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout(pad=0.2)
    save_all(fig, "fig3_ood")
    plt.close(fig)
    write_source("fig3_ood", records)


def main() -> None:
    plot_cscan()
    plot_ood()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
