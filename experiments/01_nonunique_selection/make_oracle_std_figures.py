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
    "learned-gaussian": "Learned warped (Gaussian-trained)",
    "learned-motion": "Learned warped (motion-trained)",
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


def plot_curves(ax, rows, title):
    groups = group_by_method(rows)
    order = ["plain-FBHF", "line-search-FBHF", "fixed-precond", "learned-warped-bv-c0.5", "learned-gaussian", "learned-motion"]
    for method in order:
        if method not in groups:
            continue
        label = METHOD_LABELS.get(method, method)
        g = thin_curve(groups[method])
        x = np.array([float(r["oracle"]) for r in g])
        y = np.array([float(r["mean_rel_primal_error"]) for r in g])
        lw = 1.9 if "Learned" in label else 1.35
        ax.plot(x, y, color=COLORS.get(label, "#606060"), lw=lw, alpha=0.95, label=label)
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


def write_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = [k for k in rows[0].keys() if k != "wall_time_s"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt_ms(mean, std):
    return f"{mean:.2e} +/- {std:.2e}"


def method_short(method):
    if method == "plain-FBHF":
        return "plain"
    if method == "line-search-FBHF":
        return "line-search"
    if method == "fixed-precond":
        return "fixed-warped"
    if method.startswith("learned-warped"):
        return "learned-warped"
    return method


def collect_summary(root):
    rows = []
    for sub in ["main_real_synth", "cscan_real", "ood_blur_real", "ood_noise_real", "motion_adapt"]:
        path = root / sub / "summary_formal.json"
        if path.exists():
            for r in read_json(path):
                rr = dict(r)
                rr["block"] = sub
                rows.append(rr)
    return rows


def write_tables(root, outdir):
    rows = collect_summary(root)
    table_rows = []
    for r in rows:
        table_rows.append(
            {
                "block": r["block"],
                "data": r["data"],
                "blur": r["blur"],
                "noise": r["noise"],
                "c": r["c"],
                "method": r["method"],
                "mean": r.get("err_K_mean", r["err_K"]),
                "std": r.get("err_K_std", 0.0),
                "median": r.get("err_K_median", r["err_K"]),
                "oracle_K": r.get("prox", 0) + r.get("B", 0) + r.get("C", 0),
                "network_forward_K": r.get("network", 0),
                "backtracks_K": r.get("backtracks", 0),
            }
        )
    with (outdir / "table_mean_std_all.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(table_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table_rows)

    selected = [
        ("Main real/Gaussian", "main_real_synth", "real", "train", "train", 0.5, None),
        ("Main synthetic/Gaussian", "main_real_synth", "synthetic", "train", "train", 0.5, None),
        ("c=0.25 real/Gaussian", "cscan_real", "real", "train", "train", 0.25, None),
        ("c=0.50 real/Gaussian", "cscan_real", "real", "train", "train", 0.5, None),
        ("c=0.90 real/Gaussian", "cscan_real", "real", "train", "train", 0.9, None),
        ("Small blur/high noise", "ood_blur_real", "real", "0.75", "high", 0.5, None),
        ("Wide blur/high noise", "ood_blur_real", "real", "wide", "high", 0.5, None),
        ("Motion/high, Gaussian-trained", "ood_blur_real", "real", "motion", "high", 0.5, "learned-warped-bv-c0.5"),
        ("Low noise", "ood_noise_real", "real", "train", "low", 0.5, None),
        ("High noise", "ood_noise_real", "real", "train", "high", 0.5, None),
        ("Motion/high, motion-trained", "motion_adapt", "real", "motion", "high", 0.5, "learned-warped-bv-c0.5"),
    ]
    methods = ["plain-FBHF", "line-search-FBHF", "fixed-precond"]
    md = ["| 场景 | plain | line-search | fixed warped | learned warped | learned vs fixed |", "|---|---:|---:|---:|---:|---:|"]
    compact_rows = []
    for label, block, data, blur, noise, c, _learned_filter in selected:
        pool = [
            r
            for r in rows
            if r["block"] == block
            and r["data"] == data
            and r["blur"] == blur
            and r["noise"] == noise
            and abs(float(r["c"]) - c) < 1e-12
        ]
        vals = {}
        for m in methods:
            hit = next((r for r in pool if r["method"] == m), None)
            if hit is not None:
                vals[m] = hit
        learned_hit = next((r for r in pool if r["method"].startswith("learned-warped")), None)
        if learned_hit is not None:
            vals["learned-warped"] = learned_hit
        if not vals:
            continue
        fixed = vals.get("fixed-precond")
        learned = vals.get("learned-warped")
        gain = ""
        if fixed and learned:
            gain_val = 100 * (fixed.get("err_K_mean", fixed["err_K"]) - learned.get("err_K_mean", learned["err_K"])) / fixed.get("err_K_mean", fixed["err_K"])
            gain = f"{gain_val:.1f}%"
        md.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt_ms(vals["plain-FBHF"]["err_K_mean"], vals["plain-FBHF"]["err_K_std"]) if "plain-FBHF" in vals else "",
                    fmt_ms(vals["line-search-FBHF"]["err_K_mean"], vals["line-search-FBHF"]["err_K_std"]) if "line-search-FBHF" in vals else "",
                    fmt_ms(vals["fixed-precond"]["err_K_mean"], vals["fixed-precond"]["err_K_std"]) if "fixed-precond" in vals else "",
                    fmt_ms(vals["learned-warped"]["err_K_mean"], vals["learned-warped"]["err_K_std"]) if "learned-warped" in vals else "",
                    gain,
                ]
            )
            + " |"
        )
        compact_rows.append(
            {
                "scenario": label,
                "plain_mean_std": fmt_ms(vals["plain-FBHF"]["err_K_mean"], vals["plain-FBHF"]["err_K_std"]) if "plain-FBHF" in vals else "",
                "line_search_mean_std": fmt_ms(vals["line-search-FBHF"]["err_K_mean"], vals["line-search-FBHF"]["err_K_std"]) if "line-search-FBHF" in vals else "",
                "fixed_warped_mean_std": fmt_ms(vals["fixed-precond"]["err_K_mean"], vals["fixed-precond"]["err_K_std"]) if "fixed-precond" in vals else "",
                "learned_warped_mean_std": fmt_ms(vals["learned-warped"]["err_K_mean"], vals["learned-warped"]["err_K_std"]) if "learned-warped" in vals else "",
                "learned_vs_fixed_percent": gain,
            }
        )
    (outdir / "table_mean_std_selected.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (outdir / "table_mean_std_selected.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(compact_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(compact_rows)


def main():
    root = Path(__file__).resolve().parent / "final_experiment_package" / "formal_oracle64_std_classic"
    outdir = root / "figures_oracle_meanstd"
    outdir.mkdir(parents=True, exist_ok=True)
    write_tables(root, outdir)

    main_rows = read_csv(root / "main_real_synth" / "curves_formal.csv")
    ood_rows = read_csv(root / "ood_blur_real" / "curves_formal.csv")
    motion_rows = read_csv(root / "motion_adapt" / "curves_formal.csv")
    cscan = read_json(root / "cscan_real" / "summary_formal.json")

    fig1_source = []
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=False, sharey=False)
    panel_rows = rows_for(main_rows, "real", "train", "train", 0.5)
    plot_curves(axes[0, 0], panel_rows, "a  Real classic + natural images")
    fig1_source.extend([dict(r, panel="a") for r in panel_rows])
    panel_rows = rows_for(main_rows, "synthetic", "train", "train", 0.5)
    plot_curves(axes[0, 1], panel_rows, "b  Synthetic TV images")
    fig1_source.extend([dict(r, panel="b") for r in panel_rows])
    panel_rows = rows_for(ood_rows, "real", "wide", "high", 0.5)
    plot_curves(axes[1, 0], panel_rows, "c  Strong Gaussian blur + high noise")
    fig1_source.extend([dict(r, panel="c") for r in panel_rows])
    motion_panel = []
    for r in rows_for(ood_rows, "real", "motion", "high", 0.5):
        rr = dict(r)
        if rr["method"].startswith("learned-warped"):
            rr["method"] = "learned-gaussian"
        motion_panel.append(rr)
    for r in rows_for(motion_rows, "real", "motion", "high", 0.5):
        rr = dict(r)
        if rr["method"].startswith("learned-warped"):
            rr["method"] = "learned-motion"
        elif rr["method"] in {"plain-FBHF", "line-search-FBHF", "fixed-precond"}:
            continue
        motion_panel.append(rr)
    plot_curves(axes[1, 1], motion_panel, "d  Motion blur + high noise")
    fig1_source.extend([dict(r, panel="d") for r in motion_panel])
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=1.0)
    save_all(fig, outdir / "fig1_oracle_curves_meanstd")
    write_rows(fig1_source, outdir / "source_data_fig1_oracle_curves_meanstd.csv")

    # c scan with std error bars.
    cvals = sorted({float(r["c"]) for r in cscan})
    fixed_mean, fixed_std, learned_mean, learned_std = [], [], [], []
    for c in cvals:
        pool = [r for r in cscan if abs(float(r["c"]) - c) < 1e-12]
        f = next(r for r in pool if r["method"] == "fixed-precond")
        l = next(r for r in pool if r["method"].startswith("learned-warped"))
        fixed_mean.append(f["err_K_mean"])
        fixed_std.append(f["err_K_std"])
        learned_mean.append(l["err_K_mean"])
        learned_std.append(l["err_K_std"])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.errorbar(cvals, fixed_mean, yerr=fixed_std, color="#A8A8A8", lw=1.4, marker="o", capsize=2, label="Fixed warped")
    ax.errorbar(cvals, learned_mean, yerr=learned_std, color="#B64342", lw=1.7, marker="o", capsize=2, label="Learned warped")
    ax.set_yscale("log")
    ax.set_xlabel("Bounded-variation budget c")
    ax.set_ylabel("Relative error at 4000 oracle calls")
    ax.grid(True, which="major", axis="both", lw=0.35, color="#D8D8D8", alpha=0.7)
    ax.legend()
    fig.tight_layout(pad=1.0)
    save_all(fig, outdir / "fig2_cscan_meanstd")
    with (outdir / "source_data_fig2_cscan_meanstd.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["c", "fixed_mean", "fixed_std", "learned_mean", "learned_std"])
        for row in zip(cvals, fixed_mean, fixed_std, learned_mean, learned_std):
            w.writerow(row)

    # OOD compact endpoint ratio.
    rows = collect_summary(root)
    scenarios = [
        ("small blur", "ood_blur_real", "0.75", "high", "learned-warped-bv-c0.5"),
        ("wide blur", "ood_blur_real", "wide", "high", "learned-warped-bv-c0.5"),
        ("motion\nGaussian-trained", "ood_blur_real", "motion", "high", "learned-warped-bv-c0.5"),
        ("low noise", "ood_noise_real", "train", "low", "learned-warped-bv-c0.5"),
        ("high noise", "ood_noise_real", "train", "high", "learned-warped-bv-c0.5"),
        ("motion\nmotion-trained", "motion_adapt", "motion", "high", "learned-warped-bv-c0.5"),
    ]
    ratios, line_ratios = [], []
    for _, block, blur, noise, learned_method in scenarios:
        pool = [r for r in rows if r["block"] == block and r["blur"] == blur and r["noise"] == noise and abs(float(r["c"]) - 0.5) < 1e-12]
        fixed = next(r for r in pool if r["method"] == "fixed-precond")
        learned = next(r for r in pool if r["method"] == learned_method)
        line = next(r for r in pool if r["method"] == "line-search-FBHF")
        ratios.append(learned["err_K_mean"] / fixed["err_K_mean"])
        line_ratios.append(learned["err_K_mean"] / line["err_K_mean"])
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.arange(len(scenarios))
    ax.bar(x - 0.18, ratios, width=0.34, color="#B64342", edgecolor="#272727", linewidth=0.4, label="Learned / fixed")
    ax.bar(x + 0.18, line_ratios, width=0.34, color="#E9A6A1", edgecolor="#272727", linewidth=0.4, label="Learned / line-search")
    ax.axhline(1.0, color="#606060", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in scenarios])
    ax.set_ylabel("Endpoint error ratio")
    ax.grid(True, which="major", axis="y", lw=0.35, color="#D8D8D8", alpha=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.2), ncol=2, borderaxespad=0)
    fig.tight_layout(rect=[0, 0, 1, 0.9], pad=1.0)
    save_all(fig, outdir / "fig3_ood_ratios")
    with (outdir / "source_data_fig3_ood_ratios.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "learned_over_fixed", "learned_over_line_search"])
        for s, r, lr in zip(scenarios, ratios, line_ratios):
            w.writerow([s[0], r, lr])
    print(outdir)


if __name__ == "__main__":
    main()
