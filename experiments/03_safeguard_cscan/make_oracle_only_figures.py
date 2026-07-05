import csv
import json
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
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
    }
)

COLORS = {
    "Plain FBHF": "#7884B4",
    "Line-search FBHF": "#484878",
    "Fixed warped": "#A8A8A8",
    "Learned warped": "#B64342",
    "Learned warped (Gaussian-trained)": "#E9A6A1",
    "Learned warped (motion-trained)": "#B64342",
}

METHOD_LABELS = {
    "plain-FBHF": "Plain FBHF",
    "line-search-FBHF": "Line-search FBHF",
    "fixed-precond": "Fixed warped",
    "learned-warped-bv-c0.25": "Learned warped (Gaussian-trained)",
    "learned-warped-bv-c0.5": "Learned warped (Gaussian-trained)",
    "learned-warped-bv-c0.9": "Learned warped (Gaussian-trained)",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_for(rows, data, blur, noise, c=None):
    out = []
    for r in rows:
        if r["data"] == data and r["blur"] == blur and r["noise"] == noise:
            if c is None or abs(float(r["c"]) - c) < 1e-12:
                out.append(r)
    return out


def group_by_method(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["method"], []).append(r)
    for g in groups.values():
        g.sort(key=lambda x: int(x["oracle"]))
    return groups


def thin_curve(g, max_points=220):
    if len(g) <= max_points:
        return g
    idx = np.unique(np.linspace(0, len(g) - 1, max_points).astype(int))
    return [g[i] for i in idx]


def plot_curves(ax, rows, title, rename=None):
    groups = group_by_method(rows)
    order = ["plain-FBHF", "line-search-FBHF", "fixed-precond"]
    order += [m for m in groups if m.startswith("learned-warped")]
    for method in order:
        if method not in groups:
            continue
        label = METHOD_LABELS.get(method, method)
        if rename and method in rename:
            label = rename[method]
        g = thin_curve(groups[method])
        x = np.array([float(r["oracle"]) for r in g])
        y = np.array([float(r["mean_rel_primal_error"]) for r in g])
        lw = 1.9 if "Learned" in label else 1.35
        alpha = 1.0 if "Learned" in label else 0.88
        ax.plot(x, y, color=COLORS.get(label, "#606060"), lw=lw, alpha=alpha, label=label)
    ax.set_title(title, loc="left", fontsize=8, pad=3)
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="both", lw=0.35, color="#D8D8D8", alpha=0.7)
    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("Relative primal error")


def save_all(fig, out_base):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def endpoint_table(summary):
    out = {}
    for r in summary:
        key = (r["data"], r["blur"], r["noise"], float(r["c"]), r["method"])
        out[key] = float(r["err_K"])
    return out


def write_source(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["panel", "data", "blur", "noise", "c", "method", "oracle", "network_forward", "backtracks", "mean_rel_primal_error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    root = Path(__file__).resolve().parent / "final_experiment_package" / "formal_oracle64_stream"
    outdir = root / "figures_oracle_only"
    outdir.mkdir(parents=True, exist_ok=True)

    main_rows = read_csv(root / "main_real_synth" / "curves_formal.csv")
    ood_rows = read_csv(root / "ood_real" / "curves_formal.csv")
    motion_rows = read_csv(root / "motion_adapt" / "curves_formal.csv")
    cscan_rows = read_csv(root / "cscan_real" / "curves_formal.csv")

    panels = []
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=False, sharey=False)
    plot_curves(axes[0, 0], rows_for(main_rows, "real", "train", "train", 0.5), "a  Real images, Gaussian blur")
    for r in rows_for(main_rows, "real", "train", "train", 0.5):
        r = dict(r); r["panel"] = "a"; panels.append(r)
    plot_curves(axes[0, 1], rows_for(main_rows, "synthetic", "train", "train", 0.5), "b  Synthetic TV images")
    for r in rows_for(main_rows, "synthetic", "train", "train", 0.5):
        r = dict(r); r["panel"] = "b"; panels.append(r)
    plot_curves(axes[1, 0], rows_for(ood_rows, "real", "wide", "high", 0.5), "c  Real images, stronger Gaussian blur")
    for r in rows_for(ood_rows, "real", "wide", "high", 0.5):
        r = dict(r); r["panel"] = "c"; panels.append(r)

    motion_panel = []
    for r in rows_for(ood_rows, "real", "motion", "high", 0.5):
        rr = dict(r)
        if rr["method"].startswith("learned-warped"):
            rr["method"] = "learned-warped-bv-c0.5-gaussian"
        motion_panel.append(rr)
    for r in rows_for(motion_rows, "real", "motion", "high", 0.5):
        rr = dict(r)
        if rr["method"].startswith("learned-warped"):
            rr["method"] = "learned-warped-bv-c0.5-motion"
        elif rr["method"] in {"plain-FBHF", "line-search-FBHF", "fixed-precond"}:
            continue
        motion_panel.append(rr)
    METHOD_LABELS["learned-warped-bv-c0.5-gaussian"] = "Learned warped (Gaussian-trained)"
    METHOD_LABELS["learned-warped-bv-c0.5-motion"] = "Learned warped (motion-trained)"
    plot_curves(
        axes[1, 1],
        motion_panel,
        "d  Real images, motion blur",
    )
    for r in motion_panel:
        r = dict(r); r["panel"] = "d"; panels.append(r)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=1.0)
    save_all(fig, outdir / "fig1_oracle_curves")
    write_source(panels, outdir / "source_data_fig1_oracle_curves.csv")

    # Endpoint summary: fixed-precond vs learned at K=1000.
    summaries = []
    for sub in ["main_real_synth", "ood_real", "motion_adapt"]:
        summaries.extend(read_json(root / sub / "summary_formal.json"))
    endpoint_rows = []
    conditions = [
        ("Real/Gaussian", "real", "train", "train", "main"),
        ("Synthetic/Gaussian", "synthetic", "train", "train", "main"),
        ("Real/wide high", "real", "wide", "high", "ood"),
        ("Real/motion high", "real", "motion", "high", "motion"),
    ]
    for label, data, blur, noise, source in conditions:
        pool = [r for r in summaries if r["data"] == data and r["blur"] == blur and r["noise"] == noise and abs(float(r["c"]) - 0.5) < 1e-12]
        if source == "motion":
            pool = [r for r in read_json(root / "motion_adapt" / "summary_formal.json") if r["data"] == data and r["blur"] == blur]
        fixed = next(r for r in pool if r["method"] == "fixed-precond")
        learned = next(r for r in pool if r["method"].startswith("learned-warped"))
        endpoint_rows.append((label, float(fixed["err_K"]), float(learned["err_K"])))

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(endpoint_rows))
    width = 0.34
    fixed_vals = np.array([r[1] for r in endpoint_rows])
    learned_vals = np.array([r[2] for r in endpoint_rows])
    ax.bar(x - width / 2, fixed_vals, width, color="#A8A8A8", edgecolor="#272727", linewidth=0.4, label="Fixed warped")
    ax.bar(x + width / 2, learned_vals, width, color="#B64342", edgecolor="#272727", linewidth=0.4, label="Learned warped")
    for i, (_, fv, lv) in enumerate(endpoint_rows):
        gain = 100 * (fv - lv) / fv
        label = ">99% lower" if gain > 99 else f"{gain:.0f}% lower"
        ax.text(i, min(max(fv, lv) * 1.16, 0.12), label, ha="center", va="bottom", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(7e-5, 1.8e-1)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in endpoint_rows], rotation=12, ha="right")
    ax.set_ylabel("Relative error at 4000 oracle calls")
    ax.grid(True, which="major", axis="y", lw=0.35, color="#D8D8D8", alpha=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.18), ncol=2, borderaxespad=0.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=1.0)
    save_all(fig, outdir / "fig2_endpoint_bars")
    with (outdir / "source_data_fig2_endpoint_bars.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "fixed_precond_err_K", "learned_warped_err_K", "percent_lower"])
        for label, fv, lv in endpoint_rows:
            w.writerow([label, fv, lv, 100 * (fv - lv) / fv])

    # c-scan endpoint.
    csummary = read_json(root / "cscan_real" / "summary_formal.json")
    cvals = sorted({float(r["c"]) for r in csummary})
    learned = []
    fixed = []
    for c in cvals:
        pool = [r for r in csummary if abs(float(r["c"]) - c) < 1e-12]
        fixed.append(next(float(r["err_K"]) for r in pool if r["method"] == "fixed-precond"))
        learned.append(next(float(r["err_K"]) for r in pool if r["method"].startswith("learned-warped")))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(cvals, fixed, color="#A8A8A8", lw=1.6, marker="o", label="Fixed warped")
    ax.plot(cvals, learned, color="#B64342", lw=1.9, marker="o", label="Learned warped")
    ax.set_yscale("log")
    ax.set_xlabel("Bounded-variation budget c")
    ax.set_ylabel("Relative error at 4000 oracle calls")
    ax.grid(True, which="major", axis="both", lw=0.35, color="#D8D8D8", alpha=0.7)
    ax.legend()
    fig.tight_layout(pad=1.0)
    save_all(fig, outdir / "fig3_cscan")
    with (outdir / "source_data_fig3_cscan.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["c", "fixed_precond_err_K", "learned_warped_err_K"])
        for c, fv, lv in zip(cvals, fixed, learned):
            w.writerow([c, fv, lv])

    print(outdir)


if __name__ == "__main__":
    main()
